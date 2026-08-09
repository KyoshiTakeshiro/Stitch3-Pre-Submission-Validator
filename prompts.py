"""
Prompt templates for brief evaluation.

Copied verbatim from bitcast-network/bitcast-x
(bitcast/validator/clients/prompts.py) so this tool's evaluation logic
matches what real validators run as closely as possible. Re-fetch from that
repo if validator behavior changes.

Currently supported versions: v1, v2, v3, v4 (default: v1)
"""


def generate_brief_evaluation_prompt_v1(brief, tweet):
    return (
        "///// SPONSOR BRIEF /////\n"
        f"{brief['brief']}\n\n"
        "///// TWEET /////\n"
        f"{tweet}\n\n"
        "///// YOUR TASK /////\n"
        "You are the sponsor's review agent. Decide—objectively—whether this tweet **fully** satisfies the brief.\n"
        "**Important Context**\n"
        "• The brief requirements are **minimum requirements** - creators are may choose to go deeper into the topic area - although this is not mandatory\n"
        "Additional requirement: The tweet must not be negative or critical of the sponsor.\n"
        "**Step-by-step instructions**\n\n"
        "1. **Auto-number** each requirement in the brief (1, 2, 3 …) in the order it appears.\n"
        "2. For every numbered requirement:\n"
        "   • Search the tweet.\n"
        "   • If you find evidence, mark **Met** and provide:\n"
        "       – a 3-15-word quote extracted verbatim from the tweet\n"
        "   • If no clear evidence or you are **uncertain**, mark **Not Met**.\n"
        "3. **If any item fails → Verdiction = NO.**\n\n"
        "**Important accuracy rules**\n"
        "• Do **not** invent timestamps. If a timestamp is uncertain, mark the item Not Met.\n"
        "• Fabricated quotes automatically fail that item.\n"
        "• When in doubt, choose **NO**.\n"
        "**Response format (exactly):**\n"
        "```\n"
        "## Requirement-by-Requirement\n"
        "- Req 1: [requirement text] — Met / Not Met — \"quoted evidence\" (start-sec or range)\n"
        "- Req 2: ...\n"
        "...\n"
        "## Verdict\n"
        "YES or NO\n"
        "## Summary\n"
        "Brief 1 sentence explanation of why the content did or did not meet the brief requirements.\n"
        "```\n"
        "Be concise and remember: fabricated evidence = Not Met."
    )


def generate_brief_evaluation_prompt_v2(brief, tweet):
    return (
        "///// SPONSOR BRIEF /////\n"
        f"{brief['brief']}\n\n"
        "///// TWEET /////\n"
        f"{tweet}\n\n"
        "///// YOUR TASK /////\n"
        "You are the sponsor's review agent. Decide—objectively—whether this tweet **fully** satisfies the brief.\n"
        "The brief requirements are **minimum requirements** - creators are may choose to go deeper into the topic area - although this is not mandatory\n"
        "**Base Requirements**\n"
        "• The tweet must be **predominantly (80% or more) about the sponsor or their topic** - not just a passing mention. If < 80% of the text is relevant, return NO.\n"
        "• The tweet must not be negative or critical of the sponsor\n"
        "**Step-by-step instructions**\n\n"
        "1. **Auto-number** each requirement in the brief (1, 2, 3 …) in the order it appears.\n"
        "2. For every numbered and base requirement:\n"
        "   • Search the tweet.\n"
        "   • If you find evidence, mark **Met** and provide:\n"
        "       – a 3-15-word quote extracted verbatim from the tweet\n"
        "   • If no clear evidence or you are **uncertain**, mark **Not Met**.\n"
        "3. **If any item fails → Verdiction = NO.**\n\n"
        "**Important accuracy rules**\n"
        "• Do **not** invent timestamps. If a timestamp is uncertain, mark the item Not Met.\n"
        "• Fabricated quotes automatically fail that item.\n"
        "• When in doubt, choose **NO**.\n"
        "• If the 80% relevance base requirement is Not Met, estimate what percentage of the tweet is genuinely about the sponsor/topic vs. other subject matter, and include that estimate in the Summary.\n"
        "**Response format (exactly):**\n"
        "```\n"
        "## Requirement-by-Requirement\n"
        "- Req 1: [requirement text] — Met / Not Met — \"quoted evidence\" (start-sec or range)\n"
        "- Req 2: ...\n"
        "...\n"
        "## Verdict\n"
        "YES or NO\n"
        "## Summary\n"
        "Brief 1 sentence explanation of why the content did or did not meet the brief requirements. If the 80% relevance requirement failed, state the estimated percentage breakdown (e.g. \"~40% relevant to the sponsor, 60% about other topics\").\n"
        "```\n"
        "Be concise and remember: fabricated evidence = Not Met."
    )


def generate_brief_evaluation_prompt_v4(brief, tweet):
    return (
        "///// SPONSOR BRIEF /////\n"
        f"{brief['brief']}\n\n"
        "///// TWEET /////\n"
        f"{tweet}\n\n"
        "///// YOUR TASK /////\n"
        "You are the sponsor's review agent. Decide—objectively—whether this tweet **fully** satisfies the brief.\n"
        "The brief requirements are **minimum requirements** - creators may choose to go deeper into the topic area - although this is not mandatory\n"
        "**Base Requirements**\n"
        "• The tweet must be **predominantly (80% or more) about the sponsor or their topic** - not just a passing mention. If < 80% of the text is relevant, return NO.\n"
        "**Step-by-step instructions**\n\n"
        "1. **Auto-number** each requirement in the brief (1, 2, 3 …) in the order it appears.\n"
        "2. For every numbered and base requirement:\n"
        "   • Search the tweet.\n"
        "   • If you find evidence, mark **Met** and provide:\n"
        "       – a 3-15-word quote extracted verbatim from the tweet\n"
        "   • If no clear evidence or you are **uncertain**, mark **Not Met**.\n"
        "3. **If any item fails → Verdict = NO.**\n\n"
        "**Important accuracy rules**\n"
        "• Do **not** invent timestamps. If a timestamp is uncertain, mark the item Not Met.\n"
        "• Fabricated quotes automatically fail that item.\n"
        "• When in doubt, choose **NO**.\n"
        "• If the 80% relevance base requirement is Not Met, estimate what percentage of the tweet is genuinely about the sponsor/topic vs. other subject matter, and include that estimate in the Summary.\n"
        "**Response format (exactly):**\n"
        "```\n"
        "## Requirement-by-Requirement\n"
        "- Req 1: [requirement text] — Met / Not Met — \"quoted evidence\" (start-sec or range)\n"
        "- Req 2: ...\n"
        "...\n"
        "## Verdict\n"
        "YES or NO\n"
        "## Summary\n"
        "Brief 1 sentence explanation of why the content did or did not meet the brief requirements. If the 80% relevance requirement failed, state the estimated percentage breakdown (e.g. \"~40% relevant to the sponsor, 60% about other topics\").\n"
        "```\n"
        "Be concise and remember: fabricated evidence = Not Met."
    )


def generate_brief_evaluation_prompt_v3(brief, tweet):
    backticks = "```"
    return (
        "///// TOPIC BRIEF /////\n"
        f"{brief['brief']}\n\n"
        "///// TWEET /////\n"
        f"{tweet}\n\n"
        "///// YOUR TASK /////\n"
        "Decide whether this tweet **genuinely engages** with the topic described in the brief.\n\n"
        "**Evaluation criteria**\n"
        "1. **On-topic**: The tweet must substantively address the topic — not just a passing mention or tangential reference.\n"
        "2. **Substance**: The tweet adds value — an opinion, analysis, data, comparison, prediction, or informed take on the topic.\n\n"
        "**What is allowed**\n"
        "• Critical or contrarian takes are acceptable, as long as they engage with the topic\n"
        "• Going deeper into a subtopic within the brief's scope\n\n"
        "**Step-by-step instructions**\n"
        "1. Check each evaluation criterion above.\n"
        "2. For each, mark **Met** or **Not Met** with a brief explanation.\n"
        "3. If any criterion fails → Verdict = NO.\n\n"
        "**Important accuracy rules**\n"
        "• Fabricated quotes automatically fail.\n"
        "• When in doubt, choose **NO**.\n\n"
        "**Response format (exactly):**\n"
        f"{backticks}\n"
        "## Evaluation\n"
        "- On-topic: Met / Not Met — brief explanation\n"
        "- Substance: Met / Not Met — brief explanation\n"
        "## Verdict\n"
        "YES or NO\n"
        "## Summary\n"
        "One sentence explaining why the tweet did or did not meet the brief.\n"
        f"{backticks}\n"
    )


PROMPT_GENERATORS = {
    1: generate_brief_evaluation_prompt_v1,
    2: generate_brief_evaluation_prompt_v2,
    3: generate_brief_evaluation_prompt_v3,
    4: generate_brief_evaluation_prompt_v4,
}


def get_prompt_generator(version):
    if version not in PROMPT_GENERATORS:
        raise ValueError(
            f"Unsupported prompt version: {version}. Available versions: {list(PROMPT_GENERATORS.keys())}"
        )
    return PROMPT_GENERATORS[version]


def generate_brief_evaluation_prompt(brief, tweet, version=1):
    prompt_generator = get_prompt_generator(version)
    return prompt_generator(brief, tweet)
