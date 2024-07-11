#!/bin/bash

curl -i -X POST price.buycycle.com/price_interval \
        -H "Content-Type: application/json" \
        -H "strategy: Generic" \
        -H "version: canary-001" \
        -d '[{"template_id":79204,"id":833943,"msrp":470.0,"bike_created_at_year":2024,"bike_created_at_month":6,"bike_year":null,"sales_country_id":149,"bike_type_id":2,"bike_category_id":27,"mileage_code":"less_than_500","motor":0,"condition_code":"3","brake_type_code":"hydraulic","frame_material_code":"aluminum","shifting_code":"mechanical","bike_component_id":21,"color":"#D1D5DB","family_model_id":3934,"family_id":11569,"brand_id":66,"is_mobile":0,"user_set_price":295.0,"common_price":0.0,"created_at":"2024-06-22 00:47:40","status":"active","rider_height_min":147.0,"rider_height_max":168.0,"is_ebike":0,"is_frameset":0}]'
