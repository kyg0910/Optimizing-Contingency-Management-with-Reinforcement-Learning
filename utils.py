import numpy as np
import random
import pandas as pd

def generate_data(n_sample, prize_probs, ticket_escalation, primary_substance_list, secondary_substance_list, seed=0):
    # line for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # declare basic variables
    substance_name_list = ['ALCOHOL', 'AMPHETAMINE', 'COCAINE', 'METHAMPHETAMINE', 'OPIOID']
    
    # The probability of abstaining substances before the first visit. We used empirical proportions from the first visit observations in CTN-0007 data
    prop_abstinence_init = {'ALCOHOL': 0.990, 'AMPHETAMINE': 0.933, 'COCAINE': 0.323, 
                            'METHAMPHETAMINE': 0.918, 'OPIOID': 0.559}

    # generate example CM data
    df = []
    for usubjid in range(n_sample):
        prop_abstinence_current = prop_abstinence_init.copy()
        for week in range(1, 13):
            for visit in [1, 2]:
                # save participant id, visit week, and round of visit
                current_row = {}
                current_row['USUBJID'] = usubjid
                current_row['WEEK'] = week
                current_row['ROUND OF VISIT'] = visit
                current_row = pd.DataFrame.from_dict(current_row, orient='index').T

                # save substance use results
                for substance_name in substance_name_list:
                    bernoulli_sample = np.random.binomial(n=1, p=prop_abstinence_current[substance_name], size=1)[0]
                    current_row[substance_name] = 'NEGATIVE' if bernoulli_sample else 'POSITIVE'
                    del(substance_name)
                current_row['PRIMARY SUBSTANCE'] = ('NEGATIVE' if current_row[primary_substance_list].eq('NEGATIVE').all(axis=1)[0]
                                       else 'POSITIVE')
                current_row['SECONDARY SUBSTANCE'] = ('NEGATIVE' if current_row[secondary_substance_list].eq('NEGATIVE').all(axis=1)[0]
                                                      else 'POSITIVE')
                # generate lottery results
                if (week == 1) & (visit == 1):
                    if current_row['PRIMARY SUBSTANCE'][0] == 'NEGATIVE':
                        current_row['LONGEST DURATION OF ABSTINENCE'] = 1
                        current_row['NUMBER OF DRAWS'] = ticket_escalation[0]
                        if current_row['SECONDARY SUBSTANCE'][0] == 'NEGATIVE':
                            current_row['NUMBER OF DRAWS'] += 2
                    else:
                        current_row['LONGEST DURATION OF ABSTINENCE'] = 0
                        current_row['NUMBER OF DRAWS'] = 0
                else:
                    lda_previous = np.array(df[df['USUBJID']==usubjid]['LONGEST DURATION OF ABSTINENCE'])[-1]
                    current_row['LONGEST DURATION OF ABSTINENCE'] = lda_previous
                    if current_row['PRIMARY SUBSTANCE'][0] == 'NEGATIVE':
                        if visit == 1:
                            current_row['LONGEST DURATION OF ABSTINENCE'] += 1
                        else:
                            if lda_previous == 0:
                                current_row['LONGEST DURATION OF ABSTINENCE'] = 1
                        current_row['NUMBER OF DRAWS'] = ticket_escalation[current_row['LONGEST DURATION OF ABSTINENCE'][0]-1]
                        if current_row['SECONDARY SUBSTANCE'][0] == 'NEGATIVE':
                            current_row['NUMBER OF DRAWS'] += 2
                    else:
                        current_row['LONGEST DURATION OF ABSTINENCE'] = 0
                        current_row['NUMBER OF DRAWS'] = 0
                draw_result = np.random.multinomial(current_row['NUMBER OF DRAWS'][0], prize_probs)

                current_row["NUMBER OF 'GOOD JOBS'"] = draw_result[0]
                current_row['NUMBER OF SMALL PRIZES'] = draw_result[1]
                current_row['NUMBER OF LARGE PRIZES'] = draw_result[2]
                current_row['NUMBER OF JUMBO PRIZES'] = draw_result[3]
                current_row['INCENTIVES'] = np.sum(draw_result*[0, 1, 20, 80])

                # store the current observation
                try:
                    df = pd.concat([df, current_row])
                except:
                    df = current_row

                # update the probability of abstaining substances up to the next visit
                for substance_name in substance_name_list:
                    logit_init = np.log(prop_abstinence_init[substance_name]/(1.0-prop_abstinence_init[substance_name]))
                    logit_new = logit_init + 0.1*np.log(1+current_row['INCENTIVES'][0])
                    prop_abstinence_current[substance_name] = np.clip(1.0/(1.0+np.exp(-logit_new)), 1e-12, 1.0-(1e-12))
                    del(substance_name)
                del(visit)
            del(week)
        del(usubjid)
    df = df.reset_index(drop=True)
    return df

def compute_ipw_estimate(dataframe, reward_name_list,
                         base_prize_prob_list, target_prize_prob_list,
                         base_escalation_rule, target_escalation_rule,
                         gamma=1.0,
                         base_prize_amount_list = [0, 1, 20, 80],
                         lottery_result_list = ["NUMBER OF 'GOOD JOBS'", 'NUMBER OF SMALL PRIZES',
                                                'NUMBER OF LARGE PRIZES', 'NUMBER OF JUMBO PRIZES']):
    
    target_discounted_reward_dict = {}
    target_discounted_reward_dict['usubjid'] = []
    for reward_name in reward_name_list:
        target_discounted_reward_dict[reward_name] = []
        del(reward_name)
    
    usubjid_list = np.unique(dataframe['USUBJID'])
    for usubjid in usubjid_list:
        current_df = dataframe[dataframe['USUBJID']==usubjid]
        gamma_list = [gamma**i for i in range(len(current_df))]
        reward_dict, is_weight_list = {}, []
        is_weight_list.append(1.0)
        for reward_name in reward_name_list:
            reward_dict[reward_name] = []
            del(reward_name)
        reward_dict['incentives'].append(0)
        for idx in current_df.index:
            num_of_draw_list = np.ndarray.tolist(np.array(dataframe.loc[idx, lottery_result_list]))
            prize = np.sum(np.array(num_of_draw_list)*np.array(base_prize_amount_list))
            bonus_draw = (np.array(current_df.loc[idx, ['SECONDARY SUBSTANCE']])[0] == 'NEGATIVE')
            lda = np.array(current_df.loc[idx, ['LONGEST DURATION OF ABSTINENCE']])[0]
            
            base_num_of_draw = np.sum(num_of_draw_list)
            if lda == 0:
                target_num_of_draw = 0
            else:
                target_num_of_draw = target_escalation_rule[lda-1]
                if bonus_draw:
                    target_num_of_draw += 2
            base_num_of_draw_list_list = compute_num_of_draw_list(num_of_draw=base_num_of_draw,
                                                                  prize_amount_list=base_prize_amount_list,
                                                                  total_prize=prize)
            target_num_of_draw_list_list = compute_num_of_draw_list(num_of_draw=target_num_of_draw,
                                                                    prize_amount_list=base_prize_amount_list,
                                                                    total_prize=prize)
            base_action_density = 0
            for base_num_of_draw_list in base_num_of_draw_list_list:
                base_action_density += action_density(base_num_of_draw_list,
                                                     prize_prob_list=base_prize_prob_list)
                del(base_num_of_draw_list)
            
            target_action_density = 0
            for target_num_of_draw_list in target_num_of_draw_list_list:
                target_action_density += action_density(target_num_of_draw_list,
                                                        prize_prob_list=target_prize_prob_list)
                del(target_num_of_draw_list)
            
            is_weight = (target_action_density/base_action_density)
            is_weight_list.append(is_weight)
            for reward_name in reward_name_list:
                if reward_name=='stimulant substance-free':
                    reward = 1 if (np.array(current_df.loc[idx, ['STIMULANT SUBSTANCE']])[0] == 'NEGATIVE') else 0
                elif reward_name=='alcohol-free':
                    reward = 1 if (np.array(current_df.loc[idx, ['ALCOHOL']])[0] == 'NEGATIVE') else 0
                elif reward_name=='primary substance-free':
                    reward = 1 if (np.array(current_df.loc[idx, ['PRIMARY SUBSTANCE']])[0] == 'NEGATIVE') else 0
                elif reward_name=='secondary substance-free':
                    reward = 1 if (np.array(current_df.loc[idx, ['SECONDARY SUBSTANCE']])[0] == 'NEGATIVE') else 0
                elif reward_name=='incentives':
                    reward = np.sum(np.array(current_df.loc[idx, lottery_result_list],
                                             dtype='float')
                                    *base_prize_amount_list)
                reward_dict[reward_name].append(reward)
            del(reward)
        del(idx)
        is_weight_list = is_weight_list[:(-1)]
        reward_dict['incentives'] = reward_dict['incentives'][:(-1)]
        for reward_name in reward_name_list:
            target_discounted_reward = np.sum(np.array(gamma_list)*np.array(reward_dict[reward_name])
                                        *np.array(is_weight_list))
            target_discounted_reward_dict[reward_name].append(target_discounted_reward)
        del(reward_name)
        target_discounted_reward_dict['usubjid'].append(usubjid)
    del(usubjid)
    
    return target_discounted_reward_dict

def action_density(num_of_draw_list, prize_prob_list):
    density = 1.0
    for num_of_draw, prize_prob in zip(num_of_draw_list, prize_prob_list):
        density *= (prize_prob**num_of_draw)
        del(num_of_draw, prize_prob)
    return density

def compute_num_of_draw_list(num_of_draw, prize_amount_list, total_prize):
    prize_amounts = np.array(prize_amount_list)
    solutions = []

    # Iterate through all possible combinations of draw distribution
    for combo in np.ndindex(*[num_of_draw + 1] * len(prize_amount_list)):
        if sum(combo) == num_of_draw and np.sum(prize_amounts * np.array(combo)) == total_prize:
            solutions.append(list(combo))

    return solutions
