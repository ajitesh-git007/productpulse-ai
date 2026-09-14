# Databricks notebook source
# ProductPulse AI - 04 build gold product and issue metrics
#
# Purpose:
# Create metric tables the agent can query without asking the LLM to write SQL.
# The gold layer answers trend/count questions and provides operational context
# around the retrieved textual evidence.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "adidas_retail_ai", "Unity Catalog catalog")
dbutils.widgets.text("schema", "productpulse", "Unity Catalog schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Load Bronze tables created by notebook 01. Notice that Gold is built directly
# from Bronze in this project. Silver is reserved for RAG text preparation, not
# for structured metric modeling.
products = spark.table("bronze_products")
reviews = spark.table("bronze_reviews")
tickets = spark.table("bronze_support_tickets")
orders = spark.table("bronze_orders")
returns = spark.table("bronze_returns")
inventory = spark.table("bronze_inventory_snapshots")

# Daily review aggregate. Reviews contribute both engagement volume and product
# sentiment/issue signal.
review_daily = reviews.groupBy("event_date", "product_id").agg(
    F.count("*").alias("review_count"),
    F.round(F.avg("rating"), 2).alias("avg_rating"),
    F.sum(F.when(F.col("sentiment") == "negative", 1).otherwise(0)).alias("negative_sentiment_count"),
    F.sum(F.when(F.col("issue_tag").contains("return"), 1).otherwise(0)).alias("review_return_related_count"),
)

# Daily support workload aggregate. The reason field helps connect support
# demand to return/exchange/warranty themes.
ticket_daily = tickets.groupBy("event_date", "product_id").agg(
    F.count("*").alias("ticket_count"),
    F.sum(F.when(F.col("reason").isin("return", "exchange", "warranty"), 1).otherwise(0)).alias("ticket_return_related_count"),
)

# Orders provide the denominator for return rates and the commercial demand
# context for issues. A product with many complaints but huge order volume may
# need a different action than a low-volume product with severe issues.
order_daily = orders.groupBy("event_date", "product_id").agg(
    F.countDistinct("order_id").alias("order_count"),
    F.sum("quantity").alias("units_sold"),
)

# Returns are counted separately from tickets because a ticket may ask for help
# without becoming a return, while a return can be completed without a ticket.
return_daily = returns.groupBy("event_date", "product_id").agg(
    F.count("*").alias("return_count"),
    F.sum(F.when(F.col("is_exchange"), 1).otherwise(0)).alias("exchange_count"),
)

# Inventory gives operational feasibility. Support should not recommend a size
# exchange if available stock is too low.
inventory_daily = inventory.groupBy(F.col("snapshot_date").alias("event_date"), "product_id").agg(
    F.sum("on_hand_units").alias("on_hand_units"),
    F.max("stock_status").alias("stock_status"),
)

# Pick the most common review issue for each product/day. This is a simple
# readable signal for dashboards and for the SQL metric tool.
issue_counts = (
    reviews.groupBy("event_date", "product_id", "issue_tag")
    .agg(F.count("*").alias("issue_count"))
    .withColumn(
        "issue_rank",
        F.row_number().over(
            Window.partitionBy("event_date", "product_id").orderBy(F.desc("issue_count"), F.asc("issue_tag"))
        ),
    )
    .where("issue_rank = 1")
    .select("event_date", "product_id", F.col("issue_tag").alias("top_issue"))
)

# Product daily is the main metric serving table. It combines demand, feedback,
# returns, inventory, and product attributes at one grain:
# one row per product per event_date.
gold_product_daily = (
    review_daily.join(ticket_daily, ["event_date", "product_id"], "full")
    .join(order_daily, ["event_date", "product_id"], "full")
    .join(return_daily, ["event_date", "product_id"], "full")
    .join(inventory_daily, ["event_date", "product_id"], "left")
    .join(issue_counts, ["event_date", "product_id"], "left")
    .join(products.select("product_id", "product_name", "category"), "product_id", "left")
    .fillna(
        {
            "review_count": 0,
            "ticket_count": 0,
            "order_count": 0,
            "units_sold": 0,
            "return_count": 0,
            "exchange_count": 0,
            "negative_sentiment_count": 0,
            "review_return_related_count": 0,
            "ticket_return_related_count": 0,
            "on_hand_units": 0,
            "stock_status": "unknown",
            "top_issue": "none",
        }
    )
    .withColumn(
        "return_related_count",
        F.col("review_return_related_count") + F.col("ticket_return_related_count") + F.col("return_count"),
    )
    .withColumn(
        "return_rate",
        F.when(F.col("order_count") > 0, F.round(F.col("return_count") / F.col("order_count"), 4)).otherwise(F.lit(0.0)),
    )
    .select(
        "event_date",
        "product_id",
        "product_name",
        "category",
        "review_count",
        "ticket_count",
        "order_count",
        "units_sold",
        "return_count",
        "return_rate",
        "avg_rating",
        "negative_sentiment_count",
        "return_related_count",
        "on_hand_units",
        "stock_status",
        "top_issue",
        F.current_timestamp().alias("updated_at"),
    )
)

gold_product_daily.write.mode("overwrite").saveAsTable("gold_product_daily")

# COMMAND ----------

# Gold issue daily uses a different grain:
# one row per product, date, and issue_tag. It is useful when learners want to
# extend the agent with issue-ranking or issue-trend SQL templates.
review_issues = reviews.select(
    "event_date",
    "product_id",
    "issue_tag",
    F.lit(1).alias("feedback_count"),
    F.when(F.col("sentiment") == "negative", 1).otherwise(0).alias("negative_count"),
    F.lit(0).alias("high_priority_ticket_count"),
)

# Ticket reasons become issue tags so support workload can be compared against
# review issue tags and return reasons in one table.
ticket_issues = tickets.select(
    "event_date",
    "product_id",
    F.col("reason").alias("issue_tag"),
    F.lit(1).alias("feedback_count"),
    F.when(F.col("priority") == "high", 1).otherwise(0).alias("negative_count"),
    F.when(F.col("priority") == "high", 1).otherwise(0).alias("high_priority_ticket_count"),
)

# Return reasons are treated as negative issue signals by definition because a
# return means the customer did not keep the product.
return_issues = returns.select(
    "event_date",
    "product_id",
    F.col("return_reason").alias("issue_tag"),
    F.lit(1).alias("feedback_count"),
    F.lit(1).alias("negative_count"),
    F.lit(0).alias("high_priority_ticket_count"),
)

# Union all issue-like signals and aggregate them into the issue serving table.
gold_issue_daily = (
    review_issues.unionByName(ticket_issues)
    .unionByName(return_issues)
    .groupBy("event_date", "product_id", "issue_tag")
    .agg(
        F.sum("feedback_count").alias("feedback_count"),
        F.sum("negative_count").alias("negative_count"),
        F.sum("high_priority_ticket_count").alias("high_priority_ticket_count"),
    )
    .withColumn("updated_at", F.current_timestamp())
)

gold_issue_daily.write.mode("overwrite").saveAsTable("gold_issue_daily")

print("Gold metrics build complete")
