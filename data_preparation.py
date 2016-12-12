import os
import pandas as pd
from sklearn.cross_validation import train_test_split

datadir = os.path.join(os.path.realpath('.'), 'data')
hca = pd.read_csv(os.path.join(datadir, 'ss15hca.csv'))

# Basic filtering
# Housing units only (no group quarters)
# Moved in the last year

print "Basic filtering"

filters = {
    'TYPE': 1,
    'MV': 1
}

for var in filters.keys():
    hca = hca[hca[var] == filters[var]]

print "Recoding housing variables"

# Recode tenure to binary and remove old variable

hca.loc[hca.TEN.isin([1,2]), 'tenure_own'] = 1
hca.loc[hca.TEN == 3, 'tenure_own'] = 0
hca = hca.loc[pd.notnull(hca.tenure_own)]
hca = hca.drop('TEN', axis=1)

housing_vars = [
    'ACCESS',
    'BATH',
    'RMSP',
    'YBL',
    'KIT'
]


def recode_binary(df, oldvar, newvar):
    df.loc[df[oldvar] == 1, newvar] = 1
    df.loc[df[oldvar] == 2, newvar] = 0

# Recoding categorical variables

hca.loc[hca.ACCESS.isin([1, 2]), 'access_recode'] = 1
hca.loc[hca.ACCESS == 3, 'access_recode'] = 0

recode_binary(hca, 'BATH', 'bath_recode')

hca.loc[hca.YBL.isin(range(1,7)), 'before1990'] = 1
hca.loc[hca.YBL.isin(range(7,20)), 'before1990'] = 0

recode_binary(hca, 'KIT', 'kit_recode')

housing_vars_recode = [
    'access_recode',
    'bath_recode',
    'RMSP',
    'before1990',
    'kit_recode'
]

print "Recoding household variables"

household_vars = [
    'FS',
    'LAPTOP',
    'VEH',
    'HHL',
    'HHT',
    'HINCP',
    'HUGCL',
    'HUPAC',
    'LNGI',
    'MULTG',
    'NR',
    'PARTNER',
    'SSMC'
]

recode_binary(hca, 'FS', 'fs_recode')
recode_binary(hca, 'LAPTOP', 'laptop_recode')

hca.loc[hca.HHL == 1, 'english_hh'] = 1
hca.loc[hca.HHL.isin(range(2,6)), 'english_hh'] = 0

hca.loc[hca.HHT == 1, 'single_parent'] = 0
hca.loc[hca.HHT.isin(range(4,8)), 'single_parent'] = 0
hca.loc[hca.HHT.isin(range(2,4)), 'single_parent'] = 1

hca.loc[hca.HHT == 1, 'nonfamily'] = 0
hca.loc[hca.HHT.isin(range(4,8)), 'nonfamily'] = 1
hca.loc[hca.HHT.isin(range(2,4)), 'nonfamily'] = 0

hca.loc[hca.HUPAC == 4, 'children'] = 0
hca.loc[hca.HUPAC.isin(range(1,4)), 'children'] = 1

recode_binary(hca, 'LNGI', 'good_english_speaker')
recode_binary(hca, 'MULTG', 'multigen')

hca.loc[hca.PARTNER == 0, 'unmarried_partner'] = 0
hca.loc[hca.PARTNER.isin(range(1,5)), 'unmarried_partner'] = 1

hca.loc[hca.SSMC == 0, 'samesex_marriage'] = 0
hca.loc[hca.SSMC.isin([1,2]), 'samesex_marriage'] = 1

household_vars_recode = [
    'fs_recode',
    'laptop_recode',
    'VEH',
    'english_hh',
    'single_parent',
    'nonfamily',
    'HINCP',
    'HUGCL',
    'children',
    'good_english_speaker',
    'multigen',
    'NR',
    'unmarried_partner',
    'samesex_marriage'
]

datasets = [
    hca[housing_vars + household_vars + ['tenure_own']],
    hca[household_vars + ['tenure_own']],
    hca[housing_vars_recode + household_vars_recode + ['tenure_own']],
    hca[household_vars_recode + ['tenure_own']]

]
filenames = ['hca_all.csv', 'hca_household_vars.csv', 'hca_all_recode.csv', 'hca_household_vars_recode.csv']


for index, dataset in enumerate(datasets):

    print 'Generating dataset {} with shape {}'.format(filenames[index], dataset.shape)

    dataset.to_csv(filenames[index], index=False)

    if index == 3:

        print 'Generating training and test splits:'
        df_train, df_test = train_test_split(dataset, train_size=0.7)

        print 'Full dataset shape: {}'.format(dataset.shape)
        print 'Training dataset shape: {}'.format(df_train.shape)
        print 'Test dataset shape: {}'.format(df_test.shape)

        df_train.to_csv('train_data.csv', index=False)
        df_test.to_csv('test_data.csv', index=False)
