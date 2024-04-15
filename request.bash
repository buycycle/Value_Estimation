#!/bin/bash
url="https://price.buycycle.com/price_interval"
data_template='[{"user_id": 123, "distinct_id": "abc123", "bike_id": 456, "family_id": 12, "msrp": 1200}, {}]'
# Total number of requests to send
total_requests=10
# Initialize an associative array to hold the count of HTTP status codes and versions
declare -A status_code_count
declare -A version_count
# Initialize total time and mean response time
total_time=0
mean_response_time=0
# Print initial information
current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Date and Time: $current_date_time"
echo "Address: $url"
echo "Request Payload: $data_template"
echo "Total requests to be sent: $total_requests"
# Loop total_requests times
for i in $(seq 1 $total_requests); do
  # Start the timer
  start_time=$(date +%s%N)
  # Execute the curl command and capture the headers and body
  response=$(curl -s -i -X POST "$url" \
    -H "Content-Type: application/json" \
    -H "strategy: Generic" \
    -H "model: price-dev" \
    -d "$data_template")
  # End the timer
  end_time=$(date +%s%N)
  # Calculate the time taken for this request
  elapsed_time=$(($end_time - $start_time))
  # Accumulate the total time
  total_time=$(($total_time + $elapsed_time))
  # Extract the HTTP status code and version from the response
  http_code=$(echo "$response" | grep HTTP | tail -1 | awk '{print $2}')
  version=$(echo "$response" | grep -i "^version:" | awk '{print $2}' | tr -d '\r')
  # Increment the count for this HTTP status code and version
  ((status_code_count[$http_code]++))
  ((version_count[$version]++))
done
# Calculate the mean response time in milliseconds
mean_response_time_ms=$(bc <<< "scale=2; $total_time / $total_requests / 1000000")
# Print the count of each HTTP status code
echo "HTTP Status Code Counts:"
for code in "${!status_code_count[@]}"; do
  echo "Status Code $code: ${status_code_count[$code]}"
done
# Print the count of each version
echo "Version Counts:"
for ver in "${!version_count[@]}"; do
  echo "Version $ver: ${version_count[$ver]}"
done
# Print the mean response time
echo "Mean Response Time: $mean_response_time_ms ms"
# Print completion information
echo "Total requests sent: $total_requests"

