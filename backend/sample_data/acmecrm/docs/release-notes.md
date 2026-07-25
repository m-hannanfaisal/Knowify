# AcmeCRM Release Notes

## v3.4.0 — 2026-05-12
- Webhook deliveries now retry up to 5 times with exponential backoff (previously 3).
- Added `deal.stage_changed` as a webhook event type.
- Fixed an issue where the "Send Test Event" button used a stale signing secret after regeneration.

## v3.3.0 — 2026-03-02
- Custom fields limit raised from 10 to 20 on Professional plan.
- Added CSV export scheduling for Enterprise plan (daily/weekly/monthly).
- Contacts import now supports `.xlsx` in addition to `.csv`.

## v3.2.1 — 2026-01-20
- Fixed a bug causing API rate limit headers to show incorrect remaining counts.
- Minor performance improvements to the Deals pipeline view for workspaces with 10,000+ deals.

## v3.2.0 — 2025-12-05
- Introduced API access and webhooks for Professional plan (previously Enterprise-only).
- Added SSO support for Enterprise plan (SAML 2.0).

## v3.1.0 — 2025-10-14
- Gmail and Outlook two-way email sync released (previously one-way, sent-only).
- Added Slack notification action to Automations.
