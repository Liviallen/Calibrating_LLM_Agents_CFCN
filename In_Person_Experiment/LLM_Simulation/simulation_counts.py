"""Editable total numbers of independent four-consumer simulation groups."""

DEFAULT_GROUP_COUNTS = {'Q50': 425, 'Q00': 200}
SIMULATION_CONDITIONS = tuple(DEFAULT_GROUP_COUNTS)


def groups_for_session(total_groups, sessions, session):
    """Split the total exactly, placing one extra group in earlier sessions."""
    if total_groups < 1 or sessions < 1 or not 1 <= session <= sessions:
        raise ValueError('Positive group/session counts and a valid session are required.')
    quotient, remainder = divmod(total_groups, sessions)
    return quotient + int(session <= remainder)
