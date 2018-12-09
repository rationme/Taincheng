import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True

op_train = pd.read_csv('input/operation_train_new.csv')
trans_train = pd.read_csv('input/transaction_train_new.csv')

op_test = pd.read_csv('input/operation_round1_new.csv')
trans_test = pd.read_csv('input/transaction_round1_new.csv')
y = pd.read_csv('input/tag_train_new.csv')
sub = pd.read_csv('input/sub.csv')


def get_feature(op,trans,label):
    for feature in op.columns[:]:
        if feature not in ['day']:
            if feature != 'UID':
                label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for deliver in ['ip1','mac1','mac2','geo_code']:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp

        else:
            print(feature)
            label =label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for deliver in ['ip1','mac1','mac2']:
                if feature not in deliver:
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    
                    
                    
    for feature in trans.columns[1:]:
        if feature not in ['trans_amt','bal','day']:
            if feature != 'UID':
                label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for deliver in ['merchant','ip1','mac1','geo_code',]:
                if feature not in deliver: 
                    if feature != 'UID':
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
            #if feature in ['merchant','code2','acc_id1','market_code','market_code']:
            #    label[feature+'_z'] = 0 
            #    label[feature+'_z'] = label[feature+'_y']/label[feature+'_x']
        else:
            print(feature)
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for deliver in ['merchant','ip1','mac1']:
                if feature not in deliver:
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    
                    
                    
    print("Done")
    return label


train = get_feature(op_train,trans_train,y).fillna(-1)
test = get_feature(op_test,trans_test,sub).fillna(-1)

train = train.drop(['Tag'],axis = 1).fillna(-1)
label = y['Tag']

test_id = test['UID']
test = test.drop(['Tag'],axis = 1).fillna(-1)


lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
    n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
    random_state=1000, n_jobs=16, min_child_weight=4, min_child_samples=5, min_split_gain=0)
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])

for index, (train_index, test_index) in enumerate(skf.split(train, label)):
    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]

    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
    sub_preds += test_pred / 5
    #print('test mean:', test_pred.mean())
    #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
print(m[1])
sub = pd.read_csv('input/sub.csv')
sub['Tag'] = sub_preds
sub.to_csv('sub/baseline_%s.csv'%str(m),index=False)



path = 'input/'
trans_train = pd.read_csv(path+'transaction_train_new.csv')
y = pd.read_csv(path+'tag_train_new.csv')
trans_train = trans_train.merge(y,on='UID',how='left')
def find_wrong(trans_train,y,feature):
    black = (trans_train.groupby([feature])['Tag'].sum()/trans_train.groupby([feature])['Tag'].count()).sort_values(ascending=False)
    tag_count = trans_train.groupby([feature])['Tag'].count().reset_index()
    black = black.reset_index()
    black = black.merge(tag_count,on=feature,how='left')
    black = black.sort_values(by = ['Tag_x','Tag_y'],ascending=False)
    return black

Test_trans = pd.read_csv(path+'transaction_round1_new.csv')
Test_tag = pd.read_csv('sub/baseline_%s.csv'%str(m)) # 测试样本
#rule_code = [  '5776870b5747e14e' ,'8b3f74a1391b5427' ,'0e90f47392008def' ,'6d55ccc689b910ee' ,'2260d61b622795fb' ,'1f72814f76a984fa' ,'c2e87787a76836e0' ,'4bca6018239c6201' ,'922720f3827ccef8' ,'2b2e7046145d9517' ,'09f911b8dc5dfc32' ,'7cc961258f4dce9c' ,'bc0213f01c5023ac' ,'0316dca8cc63cc17' ,'c988e79f00cc2dc0' ,'d0b1218bae116267' ,'72fac912326004ee' ,'00159b7cc2f1dfc8' ,'49ec5883ba0c1b0e' ,'c9c29fc3d44a1d7b' ,'33ce9c3877281764' ,'e7c929127cdefadb' ,'05bc3e22c112c8c9' ,'5cf4f55246093ccf' ,'6704d8d8d5965303' ,'4df1708c5827264d' ,'6e8b399ffe2d1e80' ,'f65104453e0b1d10' ,'1733ddb502eb3923' ,'a086f47f681ad851' ,'1d4372ca8a38cd1f' ,'29db08e2284ea103' ,'4e286438d39a6bd4' ,'54cb3985d0380ca4' ,'6b64437be7590eb0' ,'89eb97474a6cb3c6' ,'95d506c0e49a492c' ,'c17b47056178e2bb' ,'d36b25a74285bebb']

black = find_wrong(trans_train,y,'merchant')
rule_code_1 = black.sort_values(by=['Tag_x','Tag_y'],ascending=False).iloc[:50].merchant.tolist()
test_rule_uid = pd.DataFrame(Test_trans[Test_trans['merchant'].isin(rule_code_1)].UID.unique())
pred_data_rule = Test_tag.merge(test_rule_uid,left_on ='UID',right_on =0, how ='left')
pred_data_rule['Tag'][(pred_data_rule[0]>0)] = 1
pred_data_rule[['UID', 'Tag']].to_csv('sub/sub+rule.csv', index=False)
