-- wrk Lua script for /predict_raw endpoint
-- Generates random feature vectors for DeepFM inference based on actual data ranges
-- Campaign 2259 ranges

wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- Feature ranges based on campaign_2259_neg500 (train+val+test combined)
local feature_ranges = {
    {0, 0},      -- region (always 0 for campaign 2259)
    {0, 21},     -- city (0-21)
    {0, 2},      -- ad_exchange (0-2)
    {0, 50},     -- domain (0-13001, simplified to 0-50 for benchmark)
    {0, 50},     -- ad_slot_id (0-27097, simplified to 0-50 for benchmark)
    {0, 16},     -- ad_slot_width (0-16)
    {0, 10},     -- ad_slot_height (0-10)
    {0, 6},      -- ad_slot_visibility (0-6)
    {0, 0},      -- ad_slot_format (always 0 for campaign 2259)
    {0, 21},     -- creative_id (0-21)
    {0, 0},      -- advertiser_id (always 0 for campaign 2259)
    {0, 65},     -- user_tag (0-65)
    -- Dense features (normalized, typically 0.0-1.0)
    {0, 1},      -- ad_slot_floor_price
    {0, 1},      -- bidding_price
    {0, 1},      -- paying_price
}

-- Generate random features (12 sparse + 3 dense = 15 features)
request = function()
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

    local body = string.format('{"features":[%s]}',
        table.concat(features, ","))

    return wrk.format(nil, nil, nil, body)
end

-- Print latency statistics
done = function(summary, latency, requests)
    io.write("------------------------------\n")
    io.write("Latency Distribution:\n")
    io.write(string.format("  50%%: %.2f ms\n", latency:percentile(50) / 1000))
    io.write(string.format("  75%%: %.2f ms\n", latency:percentile(75) / 1000))
    io.write(string.format("  90%%: %.2f ms\n", latency:percentile(90) / 1000))
    io.write(string.format("  99%%: %.2f ms\n", latency:percentile(99) / 1000))
    io.write("------------------------------\n")
end
