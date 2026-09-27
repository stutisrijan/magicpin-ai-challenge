"""Per-trigger-kind playbooks: what a message must achieve and how to frame it.

    get_playbook(kind, scope, category_slug) -> Playbook

A Playbook tells both writers (the deterministic templates and the LLM prompt) the
goal, the hook and the judgment call for one trigger kind, which compulsion levers
to pull, the single CTA shape, and the concrete deliverable Vera produces on YES
(`offer`, which also drives action mode in replies).

Every kind in KNOWN_KINDS has a base playbook; a few are specialised per category
(e.g. recall_due reads differently for a dentist, a gym and a salon; a
chronic_refill_due outside pharmacies becomes a gentle check-in). Unknown kinds get
a generic playbook, customer-facing when scope == "customer".
"""

from __future__ import annotations

from dataclasses import replace

from app.schemas import (
    CTA_BINARY_CONFIRM,
    CTA_BINARY_YES_NO,
    CTA_MULTI_CHOICE_SLOT,
    CTA_OPEN_ENDED,
    Playbook,
)

# Alternative kind names seen in briefs / payloads -> canonical kind (mirrors facts.py).
KIND_ALIASES = {
    "research_digest_release": "research_digest",
    "category_research_digest_release": "research_digest",
    "compliance_alert": "regulation_change",
    "regulation_update": "regulation_change",
    "bridal_followup": "wedding_package_followup",
    "winback": "winback_eligible",
    "customer_winback": "customer_lapsed_hard",
    "heatwave": "weather_heatwave",
    "trend_movement": "category_trend_movement",
    "curious_ask": "curious_ask_due",
    "appointment_reminder": "appointment_tomorrow",
    "refill_due": "chronic_refill_due",
    "slot_open": "unplanned_slot_open",
}

CUSTOMER_KINDS = {
    "recall_due", "customer_lapsed_soft", "customer_lapsed_hard", "appointment_tomorrow",
    "chronic_refill_due", "trial_followup", "wedding_package_followup", "unplanned_slot_open",
}

_NO_FABRICATION = (
    "Use only numbers, names, dates, prices and sources that appear in the facts; if a detail is missing, "
    "leave it out rather than guess."
)
_CUSTOMER_RULES = (
    "Speak as the business (the signer), never mention Vera or magicpin, no medical claims or guarantees, "
    "quote prices only from the merchant's active offers, honour the customer's preferred time and language, "
    "and never guilt-trip about a gap."
)


def _pb(kind: str, family: str, goal: str, framing: str, levers: list[str], cta: str, cta_hint: str,
        offer: str, template_name: str, customer_facing: bool = False) -> Playbook:
    return Playbook(kind=kind, family=family, customer_facing=customer_facing, goal=goal, framing=framing,
                    levers=levers, cta=cta, cta_hint=cta_hint, offer=offer, template_name=template_name)


# --------------------------------------------------------------------------- base playbooks

_BASE: dict[str, Playbook] = {}


def _add(pb: Playbook) -> None:
    _BASE[pb.kind] = pb


# ---- knowledge / compliance -------------------------------------------------------------

_add(_pb(
    "research_digest", "knowledge",
    goal="Share one new, source-cited finding that matters to this merchant's own customers, and offer to "
         "turn it into something they can use.",
    framing=(
        "Hook: the publication itself (cite the source string exactly, e.g. 'JIDA Oct 2026, p.14') and the one-line "
        "finding. Anchor with the headline number and trial size if given. Judgment call: connect it to the "
        "merchant's own cohort or case-mix (e.g. their high-risk adult patients, their colour clients) and say who "
        "it does NOT apply to when the summary says so. Peer tone, zero hype. Do not: summarise the whole paper, "
        "invent a second study, add numbers not in the digest, or promise clinical outcomes."
    ),
    levers=["specificity", "curiosity", "reciprocity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to pull the key points and draft a customer-education note?",
    offer="pull the key points of the study and draft a customer-education WhatsApp note they can forward",
    template_name="vera_research_digest_v1",
))
_add(_pb(
    "regulation_change", "compliance",
    goal="Make sure the merchant knows a rule changed, by when, and exactly what to check in their own setup.",
    framing=(
        "Hook: the regulator and the change, with the source cited exactly and the effective date / deadline. "
        "Anchor with the concrete threshold (e.g. dose limit old vs new) and what passes vs fails. Judgment call: "
        "calm urgency, the deadline sets the pace; translate it into one audit step for their practice. Do not: "
        "alarm, threaten penalties that are not in the data, or give legal advice beyond the circular."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want a short audit checklist for your setup?",
    offer="send a short compliance audit checklist for their setup plus an SOP line they can file",
    template_name="vera_compliance_alert_v1",
))
_add(_pb(
    "cde_opportunity", "knowledge",
    goal="Get the merchant to reserve a relevant learning session (credits, date, fee from the data).",
    framing=(
        "Hook: the session title, organiser (source) and date/time. Anchor with credits and the fee exactly as "
        "given (member vs non-member). Judgment call: say why it fits this practice (topic vs their services or "
        "their city chapter). Do not: invent speakers, venues or registration links."
    ),
    levers=["specificity", "curiosity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to block the slot and remind you on the day?",
    offer="block the session in their calendar and send a reminder with the registration details on the day",
    template_name="vera_cde_opportunity_v1",
))
_add(_pb(
    "supply_alert", "compliance",
    goal="Get affected stock pulled and affected customers informed today, with Vera doing the legwork.",
    framing=(
        "Hook: urgent recall, molecule, batch numbers and manufacturer exactly as given, plus the source. Anchor: "
        "the risk in the alert's own words (e.g. sub-potency, no safety risk beyond X). Judgment call: urgent but "
        "not alarming; the pool to check is their chronic-prescription customer count if present, never an invented "
        "affected count. Offer the complete workflow (filtered list, customer note, replacement pickup). Do not: "
        "guess how many customers got the batch or add medical advice."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft the customer note and the replacement workflow?",
    offer="filter their repeat-prescription list for the affected batches and draft the customer WhatsApp note "
          "plus the replacement-pickup steps",
    template_name="vera_supply_alert_v1",
))
_add(_pb(
    "category_seasonal", "knowledge",
    goal="Turn a seasonal demand shift into one concrete stock / shelf / promo move this week.",
    framing=(
        "Hook: the season and the demand numbers (up and down movers) with the source. Judgment call: pick the "
        "one move that matters most (e.g. counter visibility for the risers, back shelf for the fallers) and tie it "
        "to the merchant's own offer or customer base. Do not: invent local demand numbers."
    ),
    levers=["specificity", "loss_aversion", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft the shelf list and a customer broadcast?",
    offer="draft a seasonal shelf and stock checklist plus a WhatsApp broadcast for their regulars",
    template_name="vera_category_seasonal_v1",
))
_add(_pb(
    "category_trend_movement", "knowledge",
    goal="Show a search trend that maps onto one of the merchant's services and position them on it.",
    framing=(
        "Hook: the search query and its YoY move (segment if given). Judgment call: connect it to a service or "
        "offer they already have; if they have none, suggest a catalog format clearly as a suggestion. Do not: "
        "claim the trend is local to their street unless the data says so."
    ),
    levers=["curiosity", "specificity", "loss_aversion"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to update your profile copy for this?",
    offer="draft a Google post and a profile description line positioned on the trend",
    template_name="vera_trend_movement_v1",
))

# ---- events -----------------------------------------------------------------------------

_add(_pb(
    "festival_upcoming", "event",
    goal="Get a festival-season package or post planned early, tied to the merchant's own offer.",
    framing=(
        "Hook: the festival and its date (and days to go when consistent). If the payload has no festival name, "
        "never name one: anchor on the category's seasonal beat for the coming window instead. Judgment call: far "
        "away means early planning, not a rush; pick one service/offer as the centrepiece. Do not: invent demand "
        "numbers, discounts or a festival."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft the festive package post for your review?",
    offer="draft a festive-season offer post and a WhatsApp broadcast for their review",
    template_name="vera_festival_upcoming_v1",
))
_add(_pb(
    "ipl_match_today", "event",
    goal="Help the restaurant make the right call for tonight's match, not just any promo.",
    framing=(
        "Hook: the match, venue and start time today. Judgment call (the point of this message): on a weekend / "
        "non-weeknight match the digest says covers drop as fans watch at home, so advise against a dine-in "
        "match-night promo and push the existing offer for delivery instead; on a weeknight, push a match-night "
        "combo (a catalog format can be suggested, never claimed as existing). Respect the offer's valid days. Do "
        "not: invent footfall numbers or discounts."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft the banner and story for tonight?",
    offer="draft a delivery-first banner and an Insta story for tonight's match",
    template_name="vera_ipl_match_v1",
))
_add(_pb(
    "weather_heatwave", "event",
    goal="Turn today's weather into one practical move for the business.",
    framing=(
        "Hook: the temperature / weather fact and city. Judgment call: one category-appropriate move (delivery "
        "push, hydration items, off-peak timing). Do not: invent demand numbers or health claims."
    ),
    levers=["specificity", "loss_aversion", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft a hot-weather post?",
    offer="draft a hot-weather Google post and a WhatsApp broadcast",
    template_name="vera_weather_alert_v1",
))
_add(_pb(
    "local_news_event", "event",
    goal="Flag a local event and its one practical implication for the merchant today.",
    framing=(
        "Hook: the event exactly as described in the data. Judgment call: one implication for footfall, timing or "
        "delivery, and one move. Do not: add details not in the data."
    ),
    levers=["specificity", "reciprocity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to post a quick update for customers?",
    offer="draft a short customer update post about the local event",
    template_name="vera_local_news_v1",
))
_add(_pb(
    "competitor_opened", "event",
    goal="Help the merchant defend on value, not price, against a new nearby listing.",
    framing=(
        "Hook: the new competitor with name, distance, opening date and their offer when given; with no data, "
        "'a new <business> listing opened near <locality>' and nothing more. Anchor: their offer vs the merchant's "
        "own active offer (e.g. their ₹199 vs your ₹299). Judgment call: no price war; defend with the merchant's "
        "real strengths (praised review themes, ratings, numbers above peer). Do not: invent a competitor name, "
        "distance, offer or review."
    ),
    levers=["loss_aversion", "social_proof", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft a post that leads with your strengths?",
    offer="draft a Google post that leads with their strongest reviews and active offer",
    template_name="vera_competitor_alert_v1",
))

# ---- performance ------------------------------------------------------------------------

_add(_pb(
    "perf_dip", "performance",
    goal="Make a measured drop feel real and fixable, with one concrete fix Vera drafts.",
    framing=(
        "Hook: the metric, the size of the drop and the window (and baseline). With a placeholder payload use the "
        "merchant's real 7-day delta with the right sign; if none, a real peer gap stated honestly (never claim a "
        "weekly drop that the data doesn't show). Judgment call: name only causes visible in the data (stale posts, "
        "unverified profile, no active offer, expired plan). Do not: invent causes or promise recovery numbers."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft the fix for your OK?",
    offer="draft a fresh Google post plus an offer line for their profile, ready for their OK",
    template_name="vera_perf_dip_v1",
))
_add(_pb(
    "perf_spike", "performance",
    goal="Credit what worked and help the merchant double down while the momentum lasts.",
    framing=(
        "Hook: the metric up, by how much, over what window; credit the likely driver if the data names one. "
        "Judgment call: one move that compounds it (follow-up post, convert the extra interest into bookings). "
        "Do not: invent drivers or future numbers."
    ),
    levers=["curiosity", "loss_aversion", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft a follow-up post while it's working?",
    offer="draft a follow-up Google post that rides the momentum",
    template_name="vera_perf_spike_v1",
))
_add(_pb(
    "seasonal_perf_dip", "performance",
    goal="Pre-empt panic about an expected seasonal dip and redirect effort to retention.",
    framing=(
        "Hook: the dip number, immediately reframed as the normal seasonal window for the category (cite the "
        "seasonal beat / digest). Judgment call: do not push acquisition spend now; redirect to retaining the "
        "existing members/customers (use their active count) and say when to spend instead if the digest says so. "
        "Do not: treat it as a failure or invent recovery dates."
    ),
    levers=["specificity", "reassurance", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft a retention challenge for your members?",
    offer="draft a four-week retention challenge (WhatsApp note plus poster copy) for their existing members",
    template_name="vera_seasonal_dip_v1",
))
_add(_pb(
    "milestone_reached", "performance",
    goal="Celebrate a real milestone briefly and convert it into the next compounding move.",
    framing=(
        "Hook: the milestone (or how close it is). With a placeholder payload, only 'crossed N' where N is a round "
        "threshold at or below a real metric. Judgment call: one move that compounds it (review requests from happy "
        "regulars, a thank-you post). Do not: round up, or claim a milestone the numbers don't support."
    ),
    levers=["social_proof", "curiosity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft the thank-you / review-request message?",
    offer="draft a thank-you post and a review-request WhatsApp message for their happy regulars",
    template_name="vera_milestone_v1",
))
_add(_pb(
    "review_theme_emerged", "performance",
    goal="Surface a review pattern and get a calm public response plus one operational fix going.",
    framing=(
        "Hook: how many reviews mention the theme in the window, whether it is rising, and one real quote. Judgment "
        "call: balance with what reviewers praise; propose drafted public replies and one fix. With no theme in the "
        "data, don't invent one: offer to pull the review summary, anchored on real numbers. Do not: argue with "
        "customers or blame staff."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft the replies for your OK?",
    offer="draft polite public replies to the recent reviews plus one operational fix note",
    template_name="vera_review_theme_v1",
))

# ---- lifecycle --------------------------------------------------------------------------

_add(_pb(
    "renewal_due", "lifecycle",
    goal="Get the plan renewed by showing value in the merchant's own numbers.",
    framing=(
        "Hook: days remaining on the named plan (and the renewal amount only if given). Anchor: their own 30-day "
        "views / calls / leads and anything above peer. Judgment call: loss framing when renewal is close (profile "
        "upkeep pauses on expiry); far away means a friendly value check-in, no urgency. Do not: quote a price that "
        "isn't in the data."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Reply YES and I'll set up the renewal for your confirmation.",
    offer="set up the plan renewal for their confirmation, with a one-page summary of the last 30 days",
    template_name="vera_renewal_due_v1",
))
_add(_pb(
    "winback_eligible", "lifecycle",
    goal="Bring a lapsed subscriber back with the real cost of the gap and one easy restart.",
    framing=(
        "Hook: days since the plan expired and what dropped since (performance, customers lapsed). Judgment call: "
        "loss framing with their numbers, zero guilt; the restart is one step and Vera does the first piece of "
        "work immediately. Do not: invent discounts."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to restart it and draft the first win-back message?",
    offer="reactivate profile upkeep and draft a win-back message to their lapsed customers",
    template_name="vera_winback_v1",
))
_add(_pb(
    "dormant_with_vera", "lifecycle",
    goal="Re-open a quiet merchant with one fresh number from their own profile and an easy question.",
    framing=(
        "Hook: one fresh, specific number from their own profile or customer base (not the silence itself). "
        "Judgment call: no guilt about not replying, no long pitch, one easy next step. Do not: repeat the last "
        "unanswered message or re-introduce Vera at length."
    ),
    levers=["curiosity", "reciprocity", "loss_aversion"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to draft it for you to check?",
    offer="draft a fresh Google post and a short comeback message for their customers",
    template_name="vera_dormant_checkin_v1",
))
_add(_pb(
    "gbp_unverified", "lifecycle",
    goal="Get the Google Business Profile verified with Vera walking them through the one step.",
    framing=(
        "Hook: the profile is not verified and the estimated uplift from verifying (only if given). Anchor: the "
        "verification path from the data (postcard or phone call). Judgment call: make it feel like one small step "
        "with Vera guiding. Do not: promise ranking or a timeline that isn't in the data."
    ),
    levers=["loss_aversion", "specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to walk you through it now?",
    offer="walk them through Google profile verification step by step",
    template_name="vera_gbp_verification_v1",
))

# ---- curiosity / planning ---------------------------------------------------------------

_add(_pb(
    "curious_ask_due", "curiosity",
    goal="Ask the merchant one easy question about their business this week and promise a concrete artifact back.",
    framing=(
        "Hook: this week's quick question, made easy with a grounded guess (a real trend signal or praised review "
        "theme). Reciprocity: say exactly what they get for answering (a Google post plus a WhatsApp reply draft). "
        "The question itself is the one CTA and comes last. Do not: ask several questions or pitch a product."
    ),
    levers=["asking_the_merchant", "curiosity", "reciprocity"],
    cta=CTA_OPEN_ENDED, cta_hint="Which service is most in demand this week?",
    offer="turn their answer into a Google post plus a WhatsApp reply they can paste for customer questions",
    template_name="vera_curious_ask_v1",
))
_add(_pb(
    "scheduled_recurring", "curiosity",
    goal="Keep a weekly cadence with one useful number and one easy question.",
    framing=(
        "Hook: this week's check-in with one real number from their profile or category trend. One easy question "
        "at the end, with a concrete artifact promised in return. Do not: turn it into a report dump."
    ),
    levers=["asking_the_merchant", "curiosity", "reciprocity"],
    cta=CTA_OPEN_ENDED, cta_hint="What's been most asked-for this week?",
    offer="turn their answer into a Google post plus a WhatsApp reply draft",
    template_name="vera_weekly_checkin_v1",
))
_add(_pb(
    "active_planning_intent", "planning",
    goal="The merchant asked what it would look like: deliver a concrete draft now, then one ask.",
    framing=(
        "Hook: acknowledge their exact ask in a few words and hand over a ready draft (3-4 bullet lines: format, "
        "who it is for, price anchor, launch). Build prices from their active offers or earlier conversation "
        "numbers only; anything new is marked as theirs to set. Judgment call: no qualifying questions, no "
        "'would you like'; the draft is the answer. Do not: invent named customers, offices, buildings or partners."
    ),
    levers=["effort_externalization", "specificity", "reciprocity"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to turn this into the post and the outreach message?",
    offer="turn the draft into a Google post and a WhatsApp outreach message",
    template_name="vera_planning_draft_v1",
))

# ---- customer-facing --------------------------------------------------------------------

_add(_pb(
    "recall_due", "customer", customer_facing=True,
    goal="Get the customer to book the due visit, ideally into one of the real open slots.",
    framing=(
        "Hook: the due service and date (or last visit) in the first sentence, from the business. Offer the exact "
        "slot labels when present (reply 1 / 2) and a way to suggest another time; otherwise ask for a convenient "
        "time. Price only from active offers. " + _CUSTOMER_RULES
    ),
    levers=["specificity", "effort_externalization", "loss_aversion_soft"],
    cta=CTA_MULTI_CHOICE_SLOT, cta_hint="Reply 1 for <slot 1>, 2 for <slot 2>, or tell us a time that works.",
    offer="book the customer into the chosen slot and send a reminder the day before",
    template_name="merchant_recall_reminder_v1",
))
_add(_pb(
    "customer_lapsed_soft", "customer", customer_facing=True,
    goal="Warmly invite a recently lapsed customer back with one easy step.",
    framing=(
        "Hook: a warm check-in with the real last-visit date / days since. Judgment call: no guilt, no 'we miss "
        "you' neediness; one easy way back (their usual service, an active offer, their preferred time). "
        + _CUSTOMER_RULES
    ),
    levers=["warmth", "effort_externalization", "specificity"],
    cta=CTA_BINARY_YES_NO, cta_hint="Reply YES and we'll hold a time that suits you.",
    offer="hold a convenient slot for the customer and confirm it by WhatsApp",
    template_name="merchant_lapsed_checkin_v1",
))
_add(_pb(
    "customer_lapsed_hard", "customer", customer_facing=True,
    goal="Win back a long-lapsed customer without shame, using their past goal and a no-pressure restart.",
    framing=(
        "Hook: how long it has been, normalised ('happens to most people'), then their previous focus or usual "
        "service. Offer one no-pressure restart (an active offer such as a trial, at their preferred time). "
        + _CUSTOMER_RULES
    ),
    levers=["warmth", "no_shame", "effort_externalization", "specificity"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want us to hold a spot for you? Reply YES.",
    offer="hold a no-pressure restart session for the customer at their preferred time",
    template_name="merchant_winback_v1",
))
_add(_pb(
    "appointment_tomorrow", "customer", customer_facing=True,
    goal="Confirm tomorrow's appointment (or reschedule it) with one reply.",
    framing=(
        "Hook: the appointment is tomorrow (date from the data; time only if given). Ask to confirm or share a new "
        "time. Short and clear. " + _CUSTOMER_RULES
    ),
    levers=["specificity", "commitment", "effort_externalization"],
    cta=CTA_BINARY_CONFIRM, cta_hint="Reply CONFIRM, or tell us a better time.",
    offer="confirm the appointment and send a reminder on the day",
    template_name="merchant_appointment_reminder_v1",
))
_add(_pb(
    "chronic_refill_due", "customer", customer_facing=True,
    goal="Get the regular refill confirmed and dispatched before the current stock runs out.",
    framing=(
        "Hook: which regular medicines are due and the run-out date. Anchor: saved delivery address, the "
        "merchant's real offers (delivery threshold, senior discount) without computing totals that aren't in the "
        "data. Respectful to seniors and family relays (Namaste, 'ji'). One reply to dispatch, and a way to flag a "
        "dose change. " + _CUSTOMER_RULES
    ),
    levers=["specificity", "effort_externalization", "trust"],
    cta=CTA_BINARY_CONFIRM, cta_hint="Reply CONFIRM to dispatch, or tell us if anything changed.",
    offer="pack the refill and dispatch it to the saved address once confirmed",
    template_name="merchant_refill_reminder_v1",
))
_add(_pb(
    "trial_followup", "customer", customer_facing=True,
    goal="Turn a completed trial into the next booked session.",
    framing=(
        "Hook: their trial date, then the next real session option(s). Mention the active offer that continues "
        "the journey, if any. For a parent relay, talk about the child by name. " + _CUSTOMER_RULES
    ),
    levers=["commitment", "specificity", "effort_externalization"],
    cta=CTA_MULTI_CHOICE_SLOT, cta_hint="Reply YES to book <slot>, or tell us a time that suits you.",
    offer="book the next session and confirm it by WhatsApp",
    template_name="merchant_trial_followup_v1",
))
_add(_pb(
    "wedding_package_followup", "customer", customer_facing=True,
    goal="Move a bride-to-be from trial to the next prep step on a relaxed timeline.",
    framing=(
        "Hook: the countdown to the wedding date (days only when consistent) and the next-step window from the "
        "data. Honour preferred days. Quote no package price unless it is an active offer. " + _CUSTOMER_RULES
    ),
    levers=["specificity", "anticipation", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want us to block a Saturday for your first session?",
    offer="block the first prep session on the customer's preferred day and share the plan",
    template_name="merchant_bridal_followup_v1",
))
_add(_pb(
    "unplanned_slot_open", "customer", customer_facing=True,
    goal="Fill a just-opened slot with a customer likely to want it.",
    framing=(
        "Hook: the exact open slot label. Keep it short; first to reply gets it. " + _CUSTOMER_RULES
    ),
    levers=["scarcity", "specificity", "effort_externalization"],
    cta=CTA_MULTI_CHOICE_SLOT, cta_hint="Reply YES to take <slot>.",
    offer="hold the open slot for the customer and confirm it",
    template_name="merchant_slot_open_v1",
))

# ---- generic fallbacks ------------------------------------------------------------------

_GENERIC_MERCHANT = _pb(
    "generic", "generic",
    goal="Explain why this update matters to the merchant now and offer one concrete next step.",
    framing=(
        "Hook: the most specific fact from the trigger payload, stated plainly as the reason for writing today. "
        "Support with one of the merchant's own numbers. One offer Vera can deliver. Do not: pad with generic "
        "growth advice or mention internal field names."
    ),
    levers=["specificity", "reciprocity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Want me to put together a quick plan for this?",
    offer="put together a short action plan based on this update",
    template_name="vera_generic_update_v1",
)
_GENERIC_CUSTOMER = _pb(
    "generic", "customer", customer_facing=True,
    goal="Give the customer one useful, specific update from the business and one easy reply.",
    framing="Hook: the specific reason for writing, from the business. One easy reply. " + _CUSTOMER_RULES,
    levers=["specificity", "effort_externalization"],
    cta=CTA_BINARY_YES_NO, cta_hint="Reply YES and we'll take care of it.",
    offer="follow up with the customer and book a convenient time",
    template_name="merchant_generic_update_v1",
)

KNOWN_KINDS: set[str] = set(_BASE)


# --------------------------------------------------------------------------- category specialisations

# (kind, category) -> field overrides. Framing overrides are appended to the base framing.
_SPECIAL: dict[tuple[str, str], dict[str, str]] = {
    ("research_digest", "dentists"): {
        "framing": "Dentists: clinical peer voice ('Dr. X'), technical vocabulary welcome (recall interval, caries, "
                   "IOPA); link to their patient cohort (high-risk adults, pediatric, aligner cases).",
        "offer": "pull the abstract's key points and draft a patient-education WhatsApp they can forward",
    },
    ("research_digest", "pharmacies"): {
        "framing": "Pharmacies: trustworthy-precise voice; link to their repeat / chronic-prescription customers.",
        "offer": "summarise the finding and draft a WhatsApp note for their repeat-prescription customers",
    },
    ("research_digest", "salons"): {
        "framing": "Salons: warm-practical voice; link to their colour / smoothening clients and stylists.",
        "offer": "summarise it and draft a client-facing post plus a stylist briefing line",
    },
    ("research_digest", "gyms"): {
        "framing": "Gyms: coach voice; turn it into a counter / trainer guideline for members.",
        "offer": "summarise it and draft a member-facing note plus a trainer guideline",
    },
    ("research_digest", "restaurants"): {
        "framing": "Restaurants: fellow-operator voice; turn it into a menu or listing move.",
        "offer": "summarise it and draft a menu / Google post change",
    },
    ("regulation_change", "dentists"): {
        "offer": "send a five-point X-ray and SOP audit checklist for their clinic",
    },
    ("regulation_change", "pharmacies"): {
        "offer": "send a register and dispensing audit checklist covering the last 90 days",
    },
    ("regulation_change", "restaurants"): {
        "offer": "send a short cost and compliance checklist for their packaging and billing",
    },
    ("category_seasonal", "pharmacies"): {
        "offer": "draft a counter-display and restock list plus a WhatsApp broadcast for their regulars",
    },
    ("category_seasonal", "salons"): {
        "offer": "draft a seasonal service post and a WhatsApp broadcast for their regulars",
    },
    ("category_seasonal", "gyms"): {
        "offer": "draft a seasonal class-schedule post and a member WhatsApp note",
    },
    ("category_seasonal", "restaurants"): {
        "offer": "draft a seasonal menu post and a WhatsApp broadcast",
    },
    ("festival_upcoming", "salons"): {
        "offer": "draft a festive package post built on their active offers, plus a WhatsApp broadcast",
    },
    ("festival_upcoming", "restaurants"): {
        "offer": "draft a festive set-menu / bulk-order post plus a WhatsApp broadcast",
    },
    ("festival_upcoming", "gyms"): {
        "offer": "draft a pre-festive shape-up plan post plus a comeback note to past members",
    },
    ("festival_upcoming", "pharmacies"): {
        "offer": "draft a festive-season health-check post plus a WhatsApp note for regular customers",
    },
    ("festival_upcoming", "dentists"): {
        "offer": "draft a seasonal patient post plus a WhatsApp recall note",
    },
    ("competitor_opened", "dentists"): {
        "framing": "Dentists: never disparage the other clinic; defend on clinical trust (reviews praising the "
                   "doctor, experience), not price.",
    },
    ("perf_dip", "restaurants"): {
        "offer": "draft a fresh Google post plus a delivery / dine-in offer line for their OK",
    },
    ("seasonal_perf_dip", "gyms"): {
        "offer": "draft a four-week attendance challenge (WhatsApp note plus poster copy) for their active members",
    },
    ("milestone_reached", "restaurants"): {
        "offer": "draft a thank-you post and a review-request note for their regular diners",
    },
    ("curious_ask_due", "restaurants"): {
        "framing": "Restaurants: ask which dish or order type moved most this week (covers, delivery vs dine-in).",
    },
    ("curious_ask_due", "dentists"): {
        "framing": "Dentists: ask which treatment patients asked about most this week.",
    },
    ("curious_ask_due", "pharmacies"): {
        "framing": "Pharmacies: ask which product or refill customers asked for most this week.",
    },
    ("curious_ask_due", "gyms"): {
        "framing": "Gyms: ask which class or goal members asked about most this week.",
    },
    ("active_planning_intent", "restaurants"): {
        "framing": "Restaurants: for bulk / corporate / catering ideas, the draft has an order-size pack, an order "
                   "window (clock times), delivery terms and the base price from their active offer.",
    },
    ("active_planning_intent", "gyms"): {
        "framing": "Gyms / studios: for a new programme, the draft has format (weeks, classes per week, age band "
                   "only if in the data), batch size, price anchor and launch channels.",
    },
    # customer-facing specialisations
    ("recall_due", "dentists"): {
        "framing": "Dentists: the routine check-up / cleaning recall; warm-clinical, no medical claims.",
        "offer": "book the patient into the chosen slot and send a reminder the day before",
    },
    ("recall_due", "gyms"): {
        "framing": "Gyms / studios: not a medical recall; frame it as time for their next session or class, "
                   "energetic but no pressure.",
        "offer": "hold a class spot for the member at their preferred time",
    },
    ("recall_due", "salons"): {
        "framing": "Salons: frame it as their regular appointment coming due (their usual service), warm and "
                   "practical.",
        "offer": "hold an appointment slot for the client at their preferred time",
    },
    ("recall_due", "pharmacies"): {
        "framing": "Pharmacies: frame it as their routine refill / health check, trustworthy and precise; no "
                   "medicine names unless in the data.",
        "offer": "keep the customer's routine refill ready and arrange pickup or delivery",
    },
    ("recall_due", "restaurants"): {
        "framing": "Restaurants: frame it as an invitation for their next visit or order; no pressure.",
        "offer": "reserve a table or keep their usual order ready",
    },
    ("chronic_refill_due", "dentists"): {
        "framing": "Outside pharmacies this is a gentle follow-up / check-in about their oral-care routine, not a "
                   "medicine refill: no product or medicine names, no claims.",
        "offer": "keep the patient's usual oral-care items ready for pickup or book a short check-up at a time "
                 "that suits them",
        "cta": CTA_BINARY_YES_NO,
        "template_name": "merchant_care_checkin_v1",
    },
    ("chronic_refill_due", "gyms"): {
        "framing": "Outside pharmacies this is a membership check-in: their monthly plan is up; keep it warm and "
                   "no-pressure.",
        "offer": "renew the member's monthly plan and confirm their usual slot",
        "cta": CTA_BINARY_YES_NO,
        "template_name": "merchant_membership_checkin_v1",
    },
    ("chronic_refill_due", "salons"): {
        "framing": "Outside pharmacies this is a friendly check-in about their hair / skin care routine; no product "
                   "claims.",
        "offer": "set aside the client's usual care products or book a quick visit",
        "cta": CTA_BINARY_YES_NO,
        "template_name": "merchant_care_checkin_v1",
    },
    ("chronic_refill_due", "restaurants"): {
        "framing": "Outside pharmacies this is a friendly nudge about their regular order; no pressure.",
        "offer": "keep the customer's regular order ready for pickup or delivery",
        "cta": CTA_BINARY_YES_NO,
        "template_name": "merchant_regular_order_v1",
    },
    ("customer_lapsed_hard", "gyms"): {
        "framing": "Gyms: coach voice, 'no judgment' normalising, their previous training focus, a no-commitment "
                   "restart at their preferred time.",
    },
    ("customer_lapsed_soft", "dentists"): {
        "offer": "book the patient's check-up at a time that suits them",
    },
    ("trial_followup", "pharmacies"): {
        "framing": "Pharmacies: a follow-up after the first order; offer to keep their next order ready.",
        "offer": "keep the customer's next order ready and arrange pickup or delivery",
        "cta": CTA_BINARY_YES_NO,
    },
    ("trial_followup", "gyms"): {
        "offer": "book the member's next session and confirm it by WhatsApp",
    },
}

_CATEGORY_WORDS = (
    ("dentists", ("dent", "dental", "orthodont")),
    ("salons", ("salon", "beauty", "spa", "parlour", "parlor", "hair")),
    ("restaurants", ("restaur", "cafe", "food", "dining", "kitchen", "bakery")),
    ("gyms", ("gym", "fitness", "yoga", "pilates", "crossfit")),
    ("pharmacies", ("pharm", "chemist", "medic", "drug")),
)


def canonical_kind(kind: str | None) -> str:
    k = (kind or "").strip().lower()
    return KIND_ALIASES.get(k, k) or "generic"


def category_key(slug: str | None) -> str:
    """Normalise a category slug ("dentists", "Dental Clinic", "yoga_studio" ...) to one of the five keys."""
    s = (slug or "").strip().lower()
    for key, words in _CATEGORY_WORDS:
        if any(w in s for w in words):
            return key
    return s


def get_playbook(kind: str, scope: str = "merchant", category_slug: str | None = None) -> Playbook:
    """The playbook for a trigger kind, specialised for scope and category. Never raises."""
    canon = canonical_kind(kind)
    customer_scope = (scope or "").strip().lower() == "customer"
    base = _BASE.get(canon)
    if base is None:
        pb = replace(_GENERIC_CUSTOMER if customer_scope else _GENERIC_MERCHANT, kind=canon or "generic")
        if canon and canon != "generic":
            pb.framing += f" (Unfamiliar trigger type '{canon.replace('_', ' ')}': explain it in plain words.) "
            pb.framing += _NO_FABRICATION
        return pb

    pb = replace(base, levers=list(base.levers))
    if customer_scope and not pb.customer_facing:
        # a merchant kind delivered to a customer: speak as the business about the same event
        pb.customer_facing = True
        pb.family = "customer"
        pb.template_name = "merchant_" + pb.template_name.removeprefix("vera_")
        pb.framing = (f"Customer-facing version of a {canon.replace('_', ' ')} update: one relevant, specific "
                      f"line for the customer and one easy reply. " + _CUSTOMER_RULES)
        pb.cta = CTA_BINARY_YES_NO
        pb.offer = _GENERIC_CUSTOMER.offer
    elif not customer_scope and pb.customer_facing:
        # a customer kind without a customer: tell the merchant and offer to send it on their behalf
        pb.customer_facing = False
        pb.family = "lifecycle"
        pb.template_name = "vera_" + pb.template_name.removeprefix("merchant_")
        pb.framing = (f"Merchant-facing version of a {canon.replace('_', ' ')} trigger: tell the owner which "
                      f"customer action is due and offer to send the message on their behalf. " + _NO_FABRICATION)
        pb.cta = CTA_BINARY_YES_NO
        pb.offer = f"send the {canon.replace('_', ' ')} message to the customer on the merchant's behalf"

    special = _SPECIAL.get((canon, category_key(category_slug)))
    if special and pb.customer_facing == (canon in CUSTOMER_KINDS):
        for field_name, value in special.items():
            if field_name == "framing":
                pb.framing = f"{pb.framing} {value}"
            else:
                setattr(pb, field_name, value)
    if _NO_FABRICATION not in pb.framing:
        pb.framing = f"{pb.framing} {_NO_FABRICATION}"
    return pb
