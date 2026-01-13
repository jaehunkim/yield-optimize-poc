-- wrk Lua script for /predict_batch endpoint
-- Generates batched random feature vectors for benchmarking
-- Usage: BATCH_SIZE=32 wrk -t4 -c100 -d30s -s wrk_predict_batch.lua http://localhost:3000/predict_batch

wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- Batch size from environment variable (default: 32)
local batch_size = tonumber(os.getenv("BATCH_SIZE")) or 32

-- Feature ranges based on campaign_2259_neg500
local feature_ranges = {
    {0, 0},      -- region (always 0 for campaign 2259)
    {0, 21},     -- city (0-21)
    {0, 2},      -- ad_exchange (0-2)
    {0, 50},     -- domain (simplified to 0-50)
    {0, 50},     -- ad_slot_id (simplified to 0-50)
    {0, 16},     -- ad_slot_width (0-16)
    {0, 10},     -- ad_slot_height (0-10)
    {0, 6},      -- ad_slot_visibility (0-6)
    {0, 0},      -- ad_slot_format (always 0)
    {0, 21},     -- creative_id (0-21)
    {0, 0},      -- advertiser_id (always 0)
    {0, 65},     -- user_tag (0-65)
    -- Dense features (normalized, 0.0-1.0)
    {0, 1},      -- ad_slot_floor_price
    {0, 1},      -- bidding_price
    {0, 1},      -- paying_price
}

-- Generate a single feature vector
local function generate_features()
    local features = {}
    for i, range in ipairs(feature_ranges) do
        local min_val, max_val = range[1], range[2]
        if i <= 12 then
            -- Sparse features: integers
            features[i] = tostring(math.random(min_val, max_val))
        else
            -- Dense features: floats
            features[i] = string.format("%.4f", min_val + math.random() * (max_val - min_val))
        end
    end
    return "[" .. table.concat(features, ",") .. "]"
end

-- Generate batch request
request = function()
    local batch_items = {}
    for i = 1, batch_size do
        batch_items[i] = generate_features()
    end

    local body = string.format('{"batch":[%s]}', table.concat(batch_items, ","))

    return wrk.format(nil, nil, nil, body)
end

-- Track statistics
local total_items = 0
local total_requests = 0

response = function(status, headers, body)
    if status == 200 then
        total_requests = total_requests + 1
        total_items = total_items + batch_size
    end
end

-- Print latency statistics
done = function(summary, latency, requests)
    io.write("------------------------------\n")
    io.write(string.format("Batch Configuration:\n"))
    io.write(string.format("  Batch size: %d\n", batch_size))
    io.write(string.format("  Total items processed: %d\n", total_items))
    io.write(string.format("  Effective throughput: %.2f items/sec\n",
        total_items / (summary.duration / 1000000)))
    io.write("------------------------------\n")
    io.write("Latency Distribution (per batch request):\n")
    io.write(string.format("  50%%: %.2f ms\n", latency:percentile(50) / 1000))
    io.write(string.format("  75%%: %.2f ms\n", latency:percentile(75) / 1000))
    io.write(string.format("  90%%: %.2f ms\n", latency:percentile(90) / 1000))
    io.write(string.format("  99%%: %.2f ms\n", latency:percentile(99) / 1000))
    io.write("------------------------------\n")
    io.write(string.format("Latency per item (estimated):\n"))
    io.write(string.format("  50%%: %.3f ms\n", latency:percentile(50) / 1000 / batch_size))
    io.write(string.format("  99%%: %.3f ms\n", latency:percentile(99) / 1000 / batch_size))
    io.write("------------------------------\n")
end
