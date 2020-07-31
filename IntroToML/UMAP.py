import umap

# Assume data is standarised and scaled

um = umap.UMAP()
um.fit(train_x)

train_x = um.transform(train_x)
test_x = um.transform(test_x)