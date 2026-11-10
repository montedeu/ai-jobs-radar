from datetime import date

import pytest

from research_radar.growth import entity_growth


@pytest.mark.parametrize(("new_max_previous", "min_mentions"), [(0, 20), (-1, 20), (21, 20)])
def test_bounds_the_query_is_promised_are_checked_before_it_runs(new_max_previous, min_mentions):
    with pytest.raises(ValueError, match="new_max_previous"):
        entity_growth(
            None,
            run_id=1,
            category="cs.CL",
            as_of=date(2026, 10, 31),
            min_mentions=min_mentions,
            new_max_previous=new_max_previous,
        )
