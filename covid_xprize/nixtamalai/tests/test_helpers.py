from covid_xprize.nixtamalai import helpers

def test_update_OxCGRT_tests():
    import pandas as pd
    #from covid_xprize import nixtamalai
    df = helpers.update_OxCGRT_tests()
    assert isinstance(df, pd.DataFrame)


def test_preprocess_full():
    #from covid_xprize import nixtamalai
    import pandas as pd
    df = helpers.preprocess_full()
    assert isinstance(df, pd.DataFrame)
