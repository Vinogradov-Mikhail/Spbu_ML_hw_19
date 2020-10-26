from model import AffinityPropagation
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
import pandas as pd
from alive_progress import alive_bar

def load_edges(path):
    edges_list = np.loadtxt(path).tolist()
    values = [max(edge[0], edge[1]) for edge in edges_list]

    n = int(np.max(values)) + 1
    lil = sparse.lil_matrix((n, n))
    for edge in edges_list:
        lil[edge[0], edge[1]] = 1
    return lil.tocsr()

def load_checkins(path):
    checkins_df = pd.read_csv(path, sep="\t", header=None)[[0, 4]]
    checkins_df.columns = ["user_id", "location_id"]
    return checkins_df

def get_users(checkins):
    user_ids = checkins.user_id.to_numpy()
    return np.unique(user_ids)

def metrics(ap, df_checkins):
    print('Splitting dataset...')

    users = get_users(df_checkins)

    permutation = np.random.permutation(users.size)
    users_random = users[permutation]
    df_checkins = df_checkins.loc[df_checkins.user_id.isin(users_random)]

    checkins_train, checkins_test = train_test_split(df_checkins, test_size=0.01, shuffle=True)

    print('Train prediction...')
    cluster_locations, top10_locations = ap.predict(checkins_train)

    cluster_prec = 0
    location_prec = 0
    userr_ids = np.unique(checkins_test.user_id)
    with alive_bar(len(userr_ids)) as bar:
        for user_id in userr_ids:
            bar()
            locations_all = checkins_test.loc[checkins_test.user_id == user_id, "location_id"].values
            uniq_locations = np.unique(locations_all)
            if user_id in checkins_train.user_id:
                train_top_locations = ap.get_top10_predict_for_user(cluster_locations, checkins_train, user_id)
                uniq_top = np.unique(train_top_locations)
                cluster_prec += np.intersect1d(uniq_top, uniq_locations).size
                location_prec += np.intersect1d(top10_locations, uniq_locations).size

    location_acc = location_prec / (10 * userr_ids.size)
    cluster_acc = cluster_prec / ( 10 * userr_ids.size)
    print('Location prediction accuracy =',location_acc)
    print('Cluster accuracy =', cluster_acc)
    return

def main():
    print('Loading data...')
    edges = load_edges('Dataset/edges.txt')

    df_checkins = load_checkins('Dataset/totalCheckins.txt')

    print('Data loaded')

    ap = AfinityPropagation()
    print('Starting training...')
    ap.fit(edges, 3)
    metrics(ap, df_checkins)
    return

if __name__ == "__main__":
    main()



