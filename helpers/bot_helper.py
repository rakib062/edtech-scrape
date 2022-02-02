import botometer
from tqdm import tqdm
# import util
import pandas as pd
import pickle
import os

# curdir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from bot_auth import rapidapi_key, twitter_app_auth


def check_bots_(user_names):

    bom = botometer.Botometer(wait_on_ratelimit=True,
                              botometer_api_url = 'https://botometer-pro.p.rapidapi.com',
                              rapidapi_key=rapidapi_key,
                              **twitter_app_auth)
    
    uname_lists = user_names# util.split_list_fixed_size(user_names, 2)
    results = dict()
    for i in tqdm(range(len(uname_lists))):
        unames = uname_lists[i]
        for screen_name, result in bom.check_accounts_in(unames):
            results[screen_name] = result
        
    return results

def check_bots(profile_dataframe_file, out_file=None):
    '''
        Test users in a profile dataframe
    '''
    users_df = pd.read_pickle(profile_dataframe_file)
    user_names = ['@'+u for u in list(users_df.username)]
    result = check_bots_(user_names)
    if out_file:
        pickle.dump(result, open(out_file, 'wb'))
    else:
        return result


def get_bot_profiles(profiles, bot_result, thresh=.49):
    '''
    Given profiles and result of bot detection, it returns:
        non bot users, representing users for who bot test result cap < threshold
        bot users
        users who were not found in the bot test result
        users for who caps were not found in the bot test result
    '''
    non_bot_users = []
    bot_scores = []
    not_found=[]
    no_cap=[]
    bots=[]
    for profile in profiles:
        key = '@'+profile.screen_name
        if key not in  bot_result:
            not_found.append(key)
            continue
        res = bot_result[key]
        if 'cap' in res.keys():
            if res['cap']['english'] < thresh:
                non_bot_users.append(profile)
            else:
                bots.append(key)
            bot_scores.append(res['cap']['english'])
        else:
            no_cap.append(key)
    return non_bot_users, bots, not_found,no_cap
