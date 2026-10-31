ARCHITECTURE: LLM CUSTOMER SUPPORT AGENT FOR E-COMMERCE

1. HIGH-LEVEL DESIGN
Use a hybrid: a workflow for triage (routing pattern) feeding an agent loop (ReAct-style) per conversation. Triage is deterministic enough to be a cheap classification call; resolution needs dynamic multi-step reasoning, so an agent handles it. Human-in-the-loop guardrails gate sensitive actions (large refunds, escalations).

Components:
- Triage service: classifies intent -> {refund, tracking, product_question, complaint} and assigns priority
- Agent runtime: autonomous loop with tools, per-conversation state, max-step and budget limits
- Tool layer: order_lookup, refund_process, tracking_status, knowledge_base_search, escalate_to_human, reply_to_customer
- State store: conversation state persisted per ticket (DB/Redis)
- Guardrail layer: wraps tool execution with policy checks
- Human queue: escalation handoff with full transcript + agent summary

2. TRIAGE (WORKFLOW)

async def triage(ticket):
    intent = await llm.generate(
        f"Classify into [refund, tracking, product_question, complaint].\n{ticket.body}")
    priority = "high" if intent == "complaint" else "normal"
    return route(intent, ticket)

Routing is a dict lookup to per-intent system prompts (refund agent gets policy text; product_question agent gets KB search emphasis).

3. AGENT LOOP (CORE ORCHESTRATION PSEUDOCODE)

async def run_agent(ticket, max_steps=8):
    state = load_or_create_state(ticket.id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + POLICY_TEXT},
        {"role": "user", "content": ticket.body},
    ] + state.history

    for step in range(max_steps):
        resp = await llm.generate(messages, tools=TOOL_SCHEMAS)
        messages.append(resp.as_message())

        if not resp.tool_calls:
            state.status = "resolved"
            save_state(state, messages)
            return resp.content  # final customer-facing reply

        for call in resp.tool_calls:
            result = await guarded_execute(call, state)   # guardrails here
            messages.append({"role": "tool",
                             "tool_call_id": call.id,
                             "content": result})
            save_state(state, messages)  # checkpoint every step

    return await escalate(ticket, messages, reason="max_steps_reached")

async def guarded_execute(call, state):
    # Policy guardrail: refunds above threshold require human approval
    if call.name == "process_refund":
        amount = call.arguments["amount"]
        order = await order_lookup(call.arguments["order_id"])
        if not state.verified_customer:
            return "ERROR: verify customer identity first (email + order match)"
        if amount > REFUND_LIMIT:  # e.g. $50
            req = await create_approval_request(call, state)
            return (f"ESCALATED: refund ${amount} exceeds limit. "
                    f"Human approval pending (id={req.id}). Inform the customer.")
    if call.name == "process_refund" and state.refund_issued:
        return "ERROR: refund already issued for this ticket (idempotency guard)"
    try:
        return await TOOLS[call.name].execute(**call.arguments)
    except Exception as e:
        return f"Error: {e}"  # errors as text so the LLM can recover

4. TOOL SCHEMAS (JSON-schema style)

- lookup_order(order_id?: string, email?: string)
  "Look up an order by ID, or match by customer email. Returns status, items, amounts, shipping/tracking info."
- get_tracking(tracking_number: string)
- search_knowledge_base(query: string, limit?: int)
- process_refund(order_id: string, amount: number, reason: string)
  "Issues a refund. Requires verified customer. Amounts over the policy limit are escalated automatically."
- escalate_to_human(ticket_id: string, summary: string, urgency: enum)
- send_reply(ticket_id: string, message: string)  // customer-facing messages

Tool design rules: text-in/text-out, self-contained results, truncated outputs, errors returned as strings (never thrown) so the agent can reason and retry.

5. MULTI-TURN MEMORY / STATE
Persist a ConversationState per ticket:

state = {
  ticket_id, intent, priority,
  customer_verified: bool,        # set after lookup_order email match
  order_id, refund_issued: bool,  # idempotency flags
  pending_approvals: [ids],
  history: [...],                 # full message transcript incl. tool results
  step_count, token_spend
}

On each new inbound message: load state, append user message, resume the loop (the transcript IS the working memory). Keep history truncated/summarized beyond ~20 turns to control tokens. The verified_customer and refund_issued flags live in state, not in the LLM's head—guardrails read them directly so prompt manipulation can't bypass policy.

6. GUARDRAILS
- Refund threshold: any process_refund above $50 auto-escalates via a pending human approval; the agent must tell the customer it's in progress
- Identity verification required before any account-affecting tool
- Idempotency: one refund per ticket/order
- Hard limits: max_steps=8, token budget, per-ticket tool-call cap
- Allowed-actions allowlist per intent (tracking agent can't call process_refund)
- All actions logged; approval requests include transcript + agent summary for the human reviewer
- Output filter on send_reply: block PII leakage and off-policy promises

7. ESCALATION FLOW
escalate() closes the agent loop, posts a summary (intent, actions taken, customer sentiment, suggested resolution) to the human queue, sets state.status='escalated', and pauses until the human resolves—new customer messages go straight to the human with agent-drafted suggested replies.
