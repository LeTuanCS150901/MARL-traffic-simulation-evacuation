import h5py


with h5py.File("offline_campfire_data_Astar", "r") as f:
    print("num samples", f["states"].shape[0])
    obs_dim = f["states"].shape[1]
    # action_dim = (obs_dim - 1) // 2
    # print(obs)

    obs = f['states'][0]
    print(obs)

    reward = f['rewards'][0]
    print(reward)

    acton = f['actions'][0]
    print(acton)

    id = f['group_id'][0].decode('utf-8')
    print(id)
