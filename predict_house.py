df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #loading data
df_train.head() #seeing initial some features or intial dataset
df_train.info() #print full summary of datafram it includes list of all columns with their data type and number of non- full values (nan) in each column


#train_test_split
train_set, test_set = make_test(df_train,test_size=0.2, random_state=654,strat_feat='Neighborhood')
tmp = train_set[['GrLivArea', 'TotRmsAbvGrd']].copy()
tmp.head()

scaler = StandardScaler()  # initialize a StandardScaler object (more on this later)
tmp = scaler.fit_transform(tmp)  # apply a fit and a transform method (more on this later)

tmp