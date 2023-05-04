scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_one)
x_train, x_test, y_train, y_test = train_test_split(df_scaled, test_size=0.2, shuffle=True, random_state=1)
print('X_train:\n', x_train[:5])
print('y_train:\n', y_train[:5])
print('X_test:\n', x_test[:5])
print('y_test:\n', y_test[:5])


#Question 2

# Backward stepwise regression
def get_stats(x,y):

    model=sm.OLS(y,x).fit()
    return model

selected=list(x.columns)
removed=[]
x_opt=x_scaled

while(True):
    model = get_stats(x_opt,y)
    max_pvalue = max(model.pvalues)
    if max_pvalue > 0.05:
        index = np.argmax(model.pvalues)
        x_opt = np.delete(x_opt, index, 1)
        #print(index,x_opt.shape)
        removed.append(selected[index])
        selected.remove(selected[index])
    else:
        break
print(f'selected features:{selected}\nremoved features:{removed}')

print(model.summary())



