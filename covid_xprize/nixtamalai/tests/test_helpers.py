

def test_update_OxCGRT_tests():
    import pandas as pd
    from covid_xprize import nixtamalai
    df = nixtamalai.helpers.update_OxCGRT_tests()
    assert isinstance(df, pd.DataFrame)


def test_preprocess():
    from covid_xprize import nixtamalai
    import pandas as pd
    df = nixtamalai.helpers.preprocess()
    assert isinstance(df, pd.DataFrame)
