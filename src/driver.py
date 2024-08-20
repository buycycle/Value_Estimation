import pandas as pd

target = "sales_price"
msrp_min = 250
msrp_max = 25000
target_min = 200
target_max = 25000

# data from eurostat https://ec.europa.eu/eurostat/web/hicp/database
current_year = 2024
inflation_rate = {
    2012: 2.6,
    2013: 1.3,
    2014: 0.4,
    2015: 0.1,
    2016: 0.2,
    2017: 1.6,
    2018: 1.8,
    2019: 1.4,
    2020: 0.7,
    2021: 2.9,
    2022: 9.2,
    2023: 6.4,
}


def cal_inflation_cum_factor(current_year, inflation_rate):
    df = pd.DataFrame(list(inflation_rate.items()), columns=["year", "inflation_rate"])

    # initialize the cumulative factor for the current year
    cumulative_factors = {current_year: 1.0}
    cumulative_factor = 1.0
    # Calculate the cumulative inflation factor for each year
    for y in range(current_year - 1, df["year"].min() - 1, -1):
        inflation_rate = df.loc[df["year"] == y, "inflation_rate"].values[0]
        cumulative_factor *= 1 + inflation_rate / 100
        cumulative_factors[y] = cumulative_factor

    # Convert the cumulative_factors dictionary to a DataFrame
    cumulative_df = pd.DataFrame(
        list(cumulative_factors.items()),
        columns=["year", "inflation_factor"],
    )
    return cumulative_df


cumulative_inflation_df = cal_inflation_cum_factor(current_year, inflation_rate)
# print(cal_inflation_cum_factor(2024, inflation_rate))
# 0   2024          1.000000
# 1   2023          1.064000
# 2   2022          1.161888
# 3   2021          1.195583
# 4   2020          1.203952
# 5   2019          1.220807
# 6   2018          1.242782
# 7   2017          1.262666
# 8   2016          1.265192
# 9   2015          1.266457
# 10  2014          1.271523
# 11  2013          1.288052
# 12  2012          1.321542

# The order of the pydantic BaseModel should follow the order [ categorical+features + numerical_features ]
# 8 categorical features
categorical_features = [
    # "template_id",  # 6595, too much for categorical features, not used now
    "brake_type_code",
    "frame_material_code",
    "shifting_code",
    "color",
    "bike_category_id",  # [1, 2, 4, <NA>, 26, 28, 19, 29, 27]
    "motor",  # [0, <NA>, 1]
    "sales_country_id",  # 29
    "bike_created_at_month",
]
 
# 17 numerical fetures
numerical_features = [
    "msrp",
    "condition_code",
    "bike_created_at_year",
    "rider_height_min",
    "rider_height_max",
    "sales_duration",
    "is_mobile",
    "is_ebike",
    "is_frameset",
    "mileage_code",
    "bike_type_id",  # only 1 and 2
    "bike_component_id",  # 72 components
    "family_model_id",  # 6382
    "family_id",  # 1732
    "brand_id",  # 334
    "bike_created_at_month_sin",
    "bike_created_at_month_cos",
    "bike_age", # calculated from bike_year and bike_created_at_year
]

test_query = """
                 SELECT


                bikes.id as id,

                bikes.bike_template_id as template_id,

                -- traget
                bookings.bike_price as sales_price,

                -- prices


                bikes.msrp as msrp,

                year(bikes.created_at) as bike_created_at_year,
                month(bikes.created_at) as bike_created_at_month



                FROM bikes

                join bookings on bikes.id = bookings.bike_id


                WHERE (TIMESTAMPDIFF(MONTH, bikes.created_at, NOW()) >= 2)


            """

main_query = """
                SELECT


                bikes.id as id,
                bikes.bike_template_id as template_id,

                -- traget
                booking_accountings.bike_price_currency as sales_price,

                -- prices


                bikes.msrp as msrp,


                -- temporal,only used for oversampling data
                bikes.created_at as bike_created_at, 
                -- temporal
                year(bikes.created_at) as bike_created_at_year,
                month(bikes.created_at) as bike_created_at_month,
                bikes.year as bike_year,



                -- take booking.created_at as end, bikes.created_at as start
         
                DATEDIFF(bookings.created_at,bikes.created_at) as sales_duration,


                -- spatial

                bikes.country_id as sales_country_id,

                -- categorizing
                bikes.bike_type_id as bike_type_id,
                bikes.bike_category_id as bike_category_id,
                bikes.mileage_code as mileage_code,
                bikes.motor as motor,

                bikes.condition_code as condition_code,



                bike_additional_infos.rider_height_min as rider_height_min,
                bike_additional_infos.rider_height_max as rider_height_max,


                bikes.brake_type_code as brake_type_code,
                bikes.frame_material_code as frame_material_code,
                bikes.shifting_code as shifting_code,
                bikes.bike_component_id as bike_component_id,

                -- find similarity between hex codes
                bikes.color as color,

                -- quite specific
                bikes.family_model_id as family_model_id,
                bikes.family_id as  family_id,
                bikes.brand_id as brand_id,

                -- is_mobile
                bikes.is_mobile as is_mobile,

                -- currency

                -- seller id

                -- is_frameset
                COALESCE(bike_template_additional_infos.is_ebike, 0) as is_ebike,
                COALESCE(bike_template_additional_infos.is_frameset, 0) as is_frameset




                FROM bikes

                join bookings on bikes.id = bookings.bike_id
                join booking_accountings on bookings.id = booking_accountings.booking_id


                left join bike_template_additional_infos on bikes.bike_template_id = bike_template_additional_infos.bike_template_id

                join bike_additional_infos on bikes.id = bike_additional_infos.bike_id


                WHERE bikes.status = 'sold'
                AND (bookings.status = 'paid_out' OR bookings.status = 'success' OR bookings.status = 'sell_link_confirm' OR bookings.status = 'capture' OR bookings.status = 'paid')
             """

# id, 26 features and 1 label
main_query_dtype = {
    "id": pd.Int64Dtype(),
    "template_id": pd.Int64Dtype(),
    "sales_price": pd.Float64Dtype(),
    "sales_duration": pd.Int64Dtype(),
    "msrp": pd.Float64Dtype(),
    "bike_created_at": str,
    "bike_created_at_year": pd.Int64Dtype(),
    "bike_created_at_month": pd.Int64Dtype(),
    "bike_year": pd.Int64Dtype(),
    "sales_country_id": pd.Int64Dtype(),
    "bike_type_id": pd.Int64Dtype(),
    "bike_category_id": pd.Int64Dtype(),
    "mileage_code": str,
    "motor": pd.Int64Dtype(),
    "rider_height_min": pd.Float64Dtype(),
    "rider_height_max": pd.Float64Dtype(),
    "brake_type_code": str,
    "condition_code": str,
    "frame_material_code": str,
    "shifting_code": str,
    "bike_component_id": pd.Int64Dtype(),
    "color": str,
    "family_model_id": pd.Int64Dtype(),
    "family_id": pd.Int64Dtype(),
    "brand_id": pd.Int64Dtype(),
    "is_mobile": pd.Int64Dtype(),
    "is_ebike": pd.Int64Dtype(),
    "is_frameset": pd.Int64Dtype(),
}


def generate_query(
    selection, maintable="bikes", joins="", where_clause="", order_by_clause=""
):
    SPECIAL_KEYWORDS = ["CASE", "YEAR", "MONTH", "COALESCE"]
    select_clauses = []
    dtype = {}
    for table, columns in selection.items():
        for expression, column_info in columns.items():
            # Check if the expression is a complex SQL expression
            expression_upper = expression.upper()
            alias = column_info['alias']
            datatype = column_info['datatype']
            if any(keyword in expression_upper for keyword in SPECIAL_KEYWORDS):
                # Directly use the expression without table prefix
                select_clauses.append(f"{expression} as {alias}")
            else:
                # Standard column, prepend table name
                select_clauses.append(f"{table}.{expression} as {alias}")
            dtype[alias] = datatype

    select_statement = ",\n".join(select_clauses)

    join_clauses = []
    for join in joins:
        join_clauses.append(
            f"{join['type']} {join['table2']} on {join['table1']}.{join['t1Column']} = {join['table2']}.{join['t2Column']}"
        )
    join_statement = "\n".join(join_clauses)

    query = f"""
        SELECT
        {select_statement}
        FROM {maintable}
        {join_statement}
        {where_clause}
        {order_by_clause}
        """.strip()
    return query, dtype