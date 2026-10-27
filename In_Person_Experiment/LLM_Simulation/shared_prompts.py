from experiment_runtime import selection

_FORMATION_BY_CONDITION = selection()["formation_by_condition"]

NAVIGATION_PROMPTS = selection()["navigations"]


def _format_subject_info(subject_type):
    if subject_type == 1:
        return "You are INFORMED."
    return "You are UNINFORMED."


def _format_product_info(subject_type, product_value):
    if subject_type == 1:
        if product_value == 3.5:
            return "You know that this product will be a big hit - every buyer receives $3.50."
        return "You know that this product will be a relative disaster - every buyer receives $0.00."
    return "The product will be a big hit ($3.50) or a relative disaster ($0.00)."


def _format_wait_time_note(wait_time):
    previous_orders = wait_time - 1
    return f"Note: Among the consumer(s) who arrived before you (if any), {previous_orders} order(s) had been placed."


# Q00/Q05/Q50 share a template with condition-specific consumer-type and decision sections.
_CONSUMER_TYPE_TEXT = {
    'Q00': """Consumer type. All consumers are uninformed. Consumers will not learn about the true product value until after the product is received. Before she places an order and before she learns how many other consumers have already ordered the product, all a consumer knows is that the product value is either $0 or $3.50 (each with a 50% probability), i.e., she would expect the product value to be (50% * $0) + (50% * $3.50) = $1.75 (on average).""",
    'Q05': """Consumer Type. For each product, and before any consumer places an order, the computer will also randomly determine the information every consumer has about the product value: With a probability of 5%, a consumer (for example, you) is informed about the product value. An informed consumer will be told about the true product value ($0 or $3.50) before she places an order, and she can simply compare the true product value with the waiting cost based on the delivery time quote. However, with a probability of 95%, a consumer (for example, you) is not informed about the product value. An uninformed consumer will not learn about the true product value until after the product is received. Before she places an order and before she learns how many other consumers have already ordered the product, all an uninformed consumer knows is that the product value is either $0 or $3.50 (each with a 50% probability), i.e., she would expect the product value to be (50% * $0) + (50% * $3.50) = $1.75 (on average).""",
    'Q50': """Consumer Type. For each product, and before any consumer places an order, the computer will also randomly determine the information every consumer has about the product value: With a probability of 50%, a consumer (for example, you) is informed about the product value. An informed consumer will be told about the true product value ($0 or $3.50) before she places an order, and she can simply compare the true product value with the waiting cost based on the delivery time quote. However, with a probability of 50%, a consumer (for example, you) is not informed about the product value. An uninformed consumer will not learn about the true product value until after the product is received. Before she places an order and before she learns how many other consumers have already ordered the product, all an uninformed consumer knows is that the product value is either $0 or $3.50 (each with a 50% probability), i.e., she would expect the product value to be (50% * $0) + (50% * $3.50) = $1.75 (on average).""",
}

_DECISION_SECTION_TEXT = {
    'Q00': """Your Decision (and how you earn)
At the beginning, you receive an endowment of $4 and you learn about your own consumer type. When you arrive at the market and you learn the delivery time, you need to make a decision whether you would want to place an order, or to not place an order.
PLACE NO ORDER. If you decide not to place an order, you simply keep your endowment of $4.
PLACE AN ORDER. If you decide to place an order and the delivery time is X weeks, you will keep your endowment of $4, MINUS the waiting costs, however, PLUS the value of the product (either $0 or $3.50). That is, if the true value of the product is $0 your payoff is $4 - $1X + $0. If the true value of the product is $3.50 your payoff is $4 - $1X + $3.50.""",
    'Q05': """Your Decision (and how you earn)
At the beginning, you receive an endowment of $4 and you learn about your own consumer type (you are either informed and know the value of the product, or you are uninformed). When you arrive at the market and you learn the delivery time, you need to make a decision whether you would want to place an order, or to not place an order.
PLACE NO ORDER. If you decide not to place an order, you simply keep your endowment of $4.
PLACE AN ORDER. If you decide to place an order and the delivery time is X weeks, you will keep your endowment of $4, MINUS the waiting costs, however, PLUS the value of the product (either $0 or $3.50). That is, if the true value of the product is $0 your payoff is $4 - $1X + $0. If the true value of the product is $3.50 your payoff is $4 - $1X + $3.50.""",
    'Q50': """Your Decision (and how you earn)
At the beginning, you receive an endowment of $4 and you learn about your own consumer type (you are either informed and know the value of the product, or you are uninformed). When you arrive at the market and you learn the delivery time, you need to make a decision whether you would want to place an order, or to not place an order.
PLACE NO ORDER. If you decide not to place an order, you simply keep your endowment of $4.
PLACE AN ORDER. If you decide to place an order and the delivery time is X weeks, you will keep your endowment of $4, MINUS the waiting costs, however, PLUS the value of the product (either $0 or $3.50). That is, if the true value of the product is $0 your payoff is $4 - $1X + $0. If the true value of the product is $3.50 your payoff is $4 - $1X + $3.50.""",
}

_INSTRUCTION_TEMPLATE = """You are about to participate in an experiment on decision making. The experiment is designed not to test your knowledge, but to learn about your decision making. 
Task Description
    A firm is facing a market of 4 potential consumers (you and 3 other people in this room). The firm brings a product to the market, which we label with a letter (e.g., 'A-Pod', 'B-Pod', etc.). All potential consumers arrive to the market and have the opportunity to place an order or not and, then, production and delivery of all orders starts. The 4 potential consumers arrive in a random order and at random times at the market. When a potential consumer (for example, you) arrives, the firm quotes a time (in weeks) at which the product will be delivered. Based on this information, the consumer must decide whether to place an order or not. Once all consumer orders are collected, the firm starts production. The firm can produce exactly one unit of the product per week and delivers the products to the consumers, in the sequence that the orders arrived. For example, imagine you arrive and the firm quotes you a delivery time of 2 weeks. Then you know that exactly one other consumer placed an order before you; it takes one week for the firm to deliver this consumer's order, plus one week for your order (if you decide to place one). However, you do not know how many potential consumers have arrived before you. It could be that you are the second consumer and the only consumer that arrived before you placed an order. But it could also be that you are the 4th consumer to arrive, and only 1 out of the 3 consumers that arrived before you placed an order (and 2 consumers did not place an order).
Waiting Costs. If you place an order, then every week you have to wait before receiving the order costs you $1. That is, the value of the product is reduced by $1 times the wait time.
True Product Value. Every consumer that orders eventually receives the product (after some delivery time) and gets the same value from owning the product. Sometimes (with 50% probability), the product is a big hit and has high value of $3.50; sometimes (with 50% probability), the product is a disaster and has low value of $0. For each product, before any consumer places an order, the computer will randomly determine the value that every consumer will receive if she decides to place an order.
{consumer_type}
{decision_section}
"""


# In-person instructions; period prompts and simulation rules are updated separately.
_IN_PERSON_INSTRUCTION_TEMPLATE = r"""# Instructions

You are about to participate in an experiment on decision making. The experiment is designed not to test your knowledge, but to learn about your decision making.

You will complete **4 trial rounds followed by 25 main rounds**. The practice rounds are designed to help you become familiar with the survey and **will not affect your payment**.

For completing the survey, you will receive a **base payment of HK$70.00**. You may also earn **additional bonus payments** based on your decisions in the 25 main rounds.

Your bonus will be based on **your average payoff across the 25 main rounds**. Each experimental dollar included in your payment equals one token, and each token you earn is worth **HK$12.00**. Your final payment will therefore consist of your **HK$70.00 base payment plus any bonus payment earned from the main rounds**.

## Your Task in Each Round

A firm offers a new product to a market of **4 consumers  (you and 3 other people in this room)**.

You remain with the same 3 other consumers throughout all 4 trial rounds and 25 main rounds. Your arrival order and conditions are randomly assigned again in each round.

- The 4 consumers arrive in a random order.
- Each consumer chooses **ORDER** or **DO NOT ORDER** once.
- Before deciding, you will see how many consumers arrived before you, how many ordered, and how many did not order.
- After all decisions are made, production is scheduled in one-week slots beginning at Week 0, following the order in which purchases were placed.
- Every round starts with a new product and a new **$4.00 endowment**.

## Product Value

The product has one value shared by all 4 consumers{including_you} in that round:

- **50% probability:** high value of **$3.50**
- **50% probability:** low value of **$0.00**

Before considering information from the current round, the expected product value is:

**50% × $3.50 + 50% × $0.00 = $1.75**

The product value is drawn once per round. A new value is drawn in the next round.

## Consumer Type

{consumer_type}

## Understanding the Wait Times

Before deciding, you will see:

- **Queuing time:** Production begins in Week 0. Each order requires one week to produce, and each order placed before yours adds one week before production of your own order begins. Therefore, your queuing time equals the number of earlier orders: 0–3 weeks.
- **Shipping time:** either 0 or 1 week. In each round, exactly 2 consumers have 0 shipping weeks and exactly 2 have 1 shipping week. {shipping_assignment}These conditions are randomly assigned again each round, separately from arrival order. Before assignment, each consumer has a 50% chance of each shipping time. Each consumer sees only their own shipping time before deciding.
- **Total wait time:** queuing time + shipping time.
- **Waiting cost:** total wait time × **$1.00**, paid only if you order.

For example, suppose your decision page shows: “3 consumers arrived before you: 1 ordered, 2 did not.” Your queuing time is 1 week: the earlier order is produced in the Week 0 slot and your order is produced in the Week 1 slot.

## Your Decision and Payoff

- **DO NOT ORDER:** You keep your $4.00 endowment.\
  **Payoff = $4.00**
- **ORDER:** You pay the waiting cost and receive the product value.\
  **Payoff = $4.00 − total waiting cost + product value**

Example: if total wait time is 2 weeks:

- Low-value product: **$4.00 − $2.00 + $0.00 = $2.00**
- High-value product: **$4.00 − $2.00 + $3.50 = $5.50**
"""

_IN_PERSON_CONSUMER_TYPE_TEXT = {
    "Q00": """Every consumer is **uninformed**. No consumer sees the product’s true value before deciding. The true value is revealed only after the decisions for the round are complete.""",
    "Q50": r"""In every round, the computer randomly assigns exactly 2 of the 4 consumers to be informed and the other 2 to be uninformed. Before this assignment, each consumer has a 50% chance of being informed.

- **50% probability: INFORMED.**  With a probability of 50%, a consumer (for example, you) is informed about the product value. An informed consumer will be told about the true product value ($0 or $3.50) before she places an order, and she can simply compare the true product value with the waiting cost.
- **50% probability: UNINFORMED.**  However, with a probability of 50%, a consumer (for example, you) is not informed about the product value. An uninformed consumer will not learn about the true product value until after you receive the product. All an uninformed consumer knows is that the product value is either $0 or $3.50 (with 50% probability each), i.e., she would expect the product value to be (50% \* $0) + (50% \* $3.50) = $1.75 (on average).

You will see your own information status before deciding. You will not see the information status of individual other consumers.""",
}


def _instruction_for_condition(condition):
    if condition in _IN_PERSON_CONSUMER_TYPE_TEXT:
        return _IN_PERSON_INSTRUCTION_TEMPLATE.format(
            including_you=" (including you)",
            consumer_type=_IN_PERSON_CONSUMER_TYPE_TEXT[condition],
            shipping_assignment=(
                "Among the 2 informed consumers, one has 0 shipping weeks and one has 1 shipping week. "
                "The same applies to the 2 uninformed consumers. "
                if condition == "Q50" else ""
            ),
        ) + _FORMATION_BY_CONDITION[condition]
    return _INSTRUCTION_TEMPLATE.format(
        consumer_type=_CONSUMER_TYPE_TEXT[condition],
        decision_section=_DECISION_SECTION_TEXT[condition],
    ) + _FORMATION_BY_CONDITION[condition]


def _instruction_q00():
    return _instruction_for_condition('Q00')


def _instruction_q05():
    return _instruction_for_condition('Q05')


def _instruction_q50():
    return _instruction_for_condition('Q50')


_INSTRUCTION_PROMPTS = {
    'Q00': _instruction_q00,
    'Q05': _instruction_q05,
    'Q50': _instruction_q50,
}


def get_instruction_prompt(condition):
    """Render the selected condition's instruction and formation."""
    if condition not in _INSTRUCTION_PROMPTS:
        raise ValueError(f'Unknown condition: {condition}. Choose Q00, Q05 or Q50.')
    return _INSTRUCTION_PROMPTS[condition]()


def _legacy_period_prompt(key, record_sheet, navigation_prompt=None):
    """
    Get the period prompt for a participant.

    Args:
        key: Tuple of (cohort_id, rank_id)
        record_sheet: Dictionary containing all experiment records
        navigation_prompt: Optional string for the navigation guidance line.
                          If None, nav_line is empty (validation default).

    Returns:
        str: The formatted prompt
    """
    wait_time = record_sheet[key]['wait_time']
    subject_type = record_sheet[key]['subject_type']
    product_name = record_sheet[key]['product_name']
    product_value = record_sheet[key]['product_value']

    wait_time_note = _format_wait_time_note(wait_time)
    subject_info = _format_subject_info(subject_type)
    product_info = _format_product_info(subject_type, product_value)

    nav_line = navigation_prompt if navigation_prompt is not None else ""

    period_prompt = f'''You have $4.00.
The firm has introduced the new {product_name}!
The firm offers you a wait time of {wait_time} week(s). {wait_time_note}
{subject_info}
{product_info}
{nav_line}
Given your stepwise explanation first and then put the final decision (ORDER/NOT ORDER) in <answer></answer>.
'''
    return period_prompt


def _weeks(value):
    return f"{value} {'week' if value == 1 else 'weeks'}"


def get_period_prompt(key, record_sheet, navigation_prompt=None):
    """Render counts from completed earlier decisions without exposing private information."""
    record = record_sheet[key]
    condition = record.get('treatment')
    if condition not in ('Q00', 'Q50'):
        return _legacy_period_prompt(key, record_sheet, navigation_prompt)
    cohort, rank = key
    choices = [str(record_sheet[(cohort, earlier)]['subject_choice'])
               for earlier in range(1, rank)]
    if any(choice not in ('0', '1') for choice in choices):
        raise ValueError("Earlier consumers must decide before rendering the next prompt.")
    ordered = choices.count('1')
    not_ordered = choices.count('0')
    shipping = record['shipping_time']
    if shipping not in (0, 1):
        raise ValueError("Shipping time must be zero or one week.")
    total = ordered + shipping
    if record['queuing_time'] != ordered or record['wait_time'] != total:
        raise ValueError("Recorded wait times do not match the preceding decisions.")
    informed = condition == 'Q50' and record['subject_type'] == 1
    subject_info = "You are INFORMED." if informed else "You are UNINFORMED."
    if informed:
        product_info = ("The product value is $3.5." if record['product_value'] == 3.5
                        else "The product value is $0.0.")
    else:
        product_info = "The product will be a big hit ($3.5) or a relative disaster ($0.0)."
    information_note = ("Each had a 50% chance of knowing the product value before deciding."
                        if condition == 'Q50' else "None knew the product value before deciding.")
    guidance = [_FORMATION_BY_CONDITION[condition].strip()]
    if navigation_prompt:
        guidance.append(navigation_prompt.strip())
    paragraphs = [
        "You have $4.00.",
        f"The firm has introduced the new {record['product_name']}!",
        "4 consumers (including you) are making purchase decisions in a randomly assigned sequence.",
        f"{rank - 1} consumers arrived before you: {ordered} ordered, {not_ordered} did not.",
        information_note,
        f"The firm offers you a total wait time of {_weeks(total)}.",
        f"This total wait time consists of {_weeks(ordered)} of queuing time and {_weeks(shipping)} of shipping time.",
        subject_info,
        product_info,
        *[text for text in guidance if text],
        "Given your stepwise explanation first and then put the final decision (ORDER/NOT ORDER) in <answer></answer>.",
    ]
    return "\n\n".join(paragraphs) + "\n"
