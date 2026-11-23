"""Circularity check for the social-engagement model.

Table 2 lists 'membership in social groups, attendance at community meetings, and leadership
roles' as the indicators of social engagement, while Table 3 uses 'attendance at community
meetings' and 'leadership roles' as its predictors. An index built from those indicators would
be regressed on its own components, so the model would be partly an identity.
"""

from __future__ import annotations

from typing import Any

SOCIAL_ENGAGEMENT_PREDICTORS = ("community_meeting_attendance", "leadership_role")


class CircularOutcomeError(ValueError):
    """The outcome is defined from a variable that is also used as a predictor."""


def circularity_problems(meta: dict[str, Any], predictors: tuple[str, ...] = SOCIAL_ENGAGEMENT_PREDICTORS) -> list[str]:
    section = meta.get("social_engagement")
    if not isinstance(section, dict):
        return ["social_engagement metadata section is missing"]
    components = section.get("components")
    if not components:
        return ["social_engagement.components must list the variables the index is built from, so "
                "circularity can be checked; an independently measured outcome is required"]
    clash = sorted(set(map(str, components)) & set(predictors))
    if clash:
        return [f"circular outcome: the engagement index is built from {clash}, which are also predictors. "
                "Supply an independently measured engagement outcome that excludes them."]
    return []


def assert_not_circular(meta: dict[str, Any]) -> None:
    problems = circularity_problems(meta)
    if problems:
        raise CircularOutcomeError("; ".join(problems))
