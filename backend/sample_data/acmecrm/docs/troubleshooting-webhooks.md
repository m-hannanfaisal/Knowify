# Troubleshooting: Webhook Verification Failures

If your integration is rejecting AcmeCRM webhook deliveries as invalid, work through these causes in order — they account for nearly all reported cases.

## 1. Wrong signing secret

Each webhook endpoint you register has its own unique signing secret. A common mistake is reusing the signing secret from a different webhook, or from your API key — these are unrelated values. Confirm the secret under **Settings → Webhooks → [webhook name] → Signing Secret**, and regenerate it if you're unsure which one you're using.

## 2. Hashing the wrong body

You must compute the HMAC over the **exact raw bytes** of the request body as received, before any JSON parsing. Many frameworks (Express with `body-parser`, Flask with automatic JSON parsing) parse the body before your handler runs, and if you then re-serialize the parsed object to compute the signature, whitespace or key-ordering differences will produce a different hash than AcmeCRM's, even though the data is "the same." Capture the raw body before any parsing middleware touches it.

## 3. Clock skew / stale payload rejection

AcmeCRM's payload includes a `timestamp` field. If your integration independently rejects payloads older than a threshold (a common anti-replay pattern), server clock drift between your infrastructure and AcmeCRM's can cause valid, freshly-delivered webhooks to be rejected as "too old." Ensure your servers are synced via NTP. AcmeCRM's own retry window is 5 minutes; a stricter threshold on your end increases the chance of false rejection.

## 4. Endpoint returning too slowly

If your endpoint takes longer than 10 seconds to respond, AcmeCRM treats the delivery as failed and retries — this is not a signature issue, but is frequently misdiagnosed as one because the retried payload has a new timestamp and can appear to "randomly" fail verification if your endpoint has inconsistent processing time. Acknowledge receipt immediately (return 200) and process the payload asynchronously.

## 5. Testing your verification logic

Use the **Settings → Webhooks → Send Test Event** button to trigger a real signed payload against your endpoint without waiting for a live event. This uses the same signing path as production events, so it's the most reliable way to isolate whether the problem is your verification code or something else (like a proxy/load balancer modifying the request body in transit — check for any WAF or API gateway that might be re-encoding the payload).
