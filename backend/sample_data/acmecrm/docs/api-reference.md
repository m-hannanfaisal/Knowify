# AcmeCRM API Reference

The AcmeCRM API is available on the **Professional** and **Enterprise** plans. It is not available on Starter.

Base URL: `https://api.acmecrm.com/v1`

## Authentication

All requests require an API key sent in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

### Creating an API Key

1. Log in to AcmeCRM and go to **Settings → API Keys**.
2. Click **Generate New Key**.
3. Name the key (e.g. "Zapier integration") and choose its scope: `read-only` or `read-write`.
4. Copy the key immediately — it is shown only once. If lost, you must revoke it and generate a new one.

Each workspace can have up to 10 active API keys. Keys can be revoked at any time from the same page; revoking takes effect within 60 seconds.

## Rate Limits

- Professional: 100 requests/minute
- Enterprise: 1,000 requests/minute, or custom by agreement

Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header.

## Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/contacts` | List contacts, paginated |
| POST | `/contacts` | Create a contact |
| GET | `/contacts/{id}` | Retrieve a single contact |
| PATCH | `/contacts/{id}` | Update a contact |
| DELETE | `/contacts/{id}` | Delete a contact |
| GET | `/deals` | List deals |
| POST | `/deals` | Create a deal |
| GET | `/webhooks` | List registered webhooks |
| POST | `/webhooks` | Register a new webhook |

## Webhooks

Webhooks let your application receive real-time events (e.g. `contact.created`, `deal.stage_changed`) as HTTP POST requests to a URL you specify. Available on Professional and Enterprise plans only.

### Registering a Webhook

```
POST /webhooks
{
  "url": "https://yourapp.com/hooks/acmecrm",
  "events": ["contact.created", "deal.stage_changed"]
}
```

### Verifying Webhook Signatures

Every webhook request includes an `X-AcmeCRM-Signature` header, an HMAC-SHA256 hash of the raw request body, signed using the webhook's signing secret (shown once when the webhook is created, and re-viewable under **Settings → Webhooks → [webhook name] → Signing Secret**).

To verify:
1. Compute `HMAC-SHA256(signing_secret, raw_request_body)`.
2. Compare it to the value in `X-AcmeCRM-Signature` using a constant-time comparison.
3. Reject the request if they don't match.

**Common causes of signature verification failure** (see `troubleshooting-webhooks.md` for full detail):
- Using the wrong signing secret (each webhook has its own; they are not interchangeable).
- Hashing a parsed/re-serialized body instead of the exact raw bytes received.
- Clock skew — AcmeCRM includes a timestamp in the payload and rejects requests processed more than 5 minutes after signing if your integration also checks staleness.

Webhook deliveries are retried up to 5 times with exponential backoff if your endpoint returns a non-2xx status or times out.
