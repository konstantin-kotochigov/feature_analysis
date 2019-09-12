from tqdm import tqdm
import math
import pandas

# Function to extract basic feature statistics
def get_feature_stats(df):

    total_events = df[df.target == 1].shape[0]
    total_nonevents = df[df.target == 0].shape[0]

    # Compute WOE metrics
    woe_dict = {}
    for colnum, col in tqdm(enumerate(df.columns), total=len(df.columns)):
        subgroup = df[df[col] == 1]
        subgroup_size = df[df[col] == 1].shape[0]
        if ((subgroup_size < 1) | (subgroup[subgroup.target == 1].shape[0] == 0) | (
                subgroup[subgroup.target == 0].shape[0] == 0)):
            continue
        event_rate = round(subgroup[subgroup.target == 1].shape[0] / total_events, 5)
        nonevent_rate = round(subgroup[subgroup.target == 0].shape[0] / total_nonevents, 5)
        event_precision = round(subgroup[subgroup.target == 1].shape[0] / subgroup_size, 5)
        event_num = subgroup[subgroup.target == 1].shape[0]
        nonevent_num = subgroup[subgroup.target == 0].shape[0]
        woe_dict[col] = (
        round(math.log(event_rate / nonevent_rate), 4), event_num, nonevent_num, event_rate, nonevent_rate,
        event_precision)

    # Format as a table
    woe = pandas.DataFrame.from_dict(woe_dict, orient="index",
                                     columns=['woe_value', 'event_num', 'nonevent_num', 'event_rate', 'nonevent_rate',
                                              'event_precision'])
    woe = woe.reset_index()
    woe.columns = ['attr_id', 'woe_value', 'event_num', 'nonevent_num', 'event_rate', 'nonevent_rate',
                   'event_precision']
    woe['woe_abs'] = woe.woe_value.apply(abs)
    woe.sort_values("woe_abs", ascending=False, inplace=True)
    woe.reset_index(drop=True, inplace=True)

    # Filter Rare Features
    woe = woe[woe.event_num >= 5]
    woe.sort_values("woe_abs", ascending=False, inplace=True)

    # Add dataframe-wise class balance
    woe['total_precision'] = round(total_events / df.shape[0], 5)

    return woe[['attr_id', 'attr_desc', 'event_num', 'nonevent_num', 'event_rate', 'nonevent_rate', 'event_precision',
                'total_precision', 'woe_value']]