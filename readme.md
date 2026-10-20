
# FlowSentinel

**The AI agent that knows when NOT to automate.**

FlowSentinel is an experimental agent-safety prototype built for the AWS Agent Experience Hackathon: Strands Decider.

Amazon Bedrock proposes tool calls. Strands Decider 2B evaluates whether the proposed arguments are grounded in the user's request and whether clarification is needed. A Strands intervention uses those typed decisions to let the tool proceed or guide the agent to correct its proposal.

## Demonstrated behavior

- A clear task request proceeds.
- A missing task title is blocked until clarified.
- An invented task title is blocked, after which the agent can recover and submit a grounded request.

The current demo uses a simulated task tool. It does not modify external systems.

## Architecture

1. User provides a request.
2. Amazon Bedrock proposes a tool call.
3. `GroundedToolGuard` sends the conversation and proposed arguments to the local Decider server.
4. Decider returns typed probabilities.
5. The intervention returns `Proceed` or `Guide`.
6. The agent either executes the tool or corrects its proposal.

## Run locally

Requirements: Python, AWS credentials with Amazon Bedrock access, and the Strands Decider model downloaded locally.

Start the decision server in one terminal:

```powershell
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu --port 8099
```

In another terminal, activate your virtual environment, install dependencies, and run:

```powershell
python -m pip install -r requirements.txt
python app.py
```

The default demo uses `amazon.nova-lite-v1:0`. Ensure that model is available to your AWS account in the configured region.

## Reusable decision pattern

`GroundedToolGuard` demonstrates a reusable pre-tool-call guardrail using typed yes/no questions:

- Are the proposed argument values grounded in the user's request?
- Does the action need clarification?

The policy maps these probabilities to Strands intervention actions. The threshold is experimental and should be evaluated against representative inputs before reuse.

## Limitations

- Decider 2B is experimental.
- Confidence is not a guarantee of correctness.
- This guardrail is not a substitute for authorization, permissions, or deterministic validation.
- The task tool is simulated.

## Hackathon tags

#strands-decider-2b #guardrails