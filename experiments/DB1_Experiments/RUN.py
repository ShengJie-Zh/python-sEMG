import statsmodels.formula.api as smf
import numpy as np
from ..utils.features import *

result_enhanced = []
result_orginal = []


for feat in range(0,42):
    flag = 0
    if star!=0:
        accuracy = np.concatenate([result_orginal, result_enhanced])
        method = ['Raw'] * 20 + ['Enhanced'] * 20
        subject_ids = np.repeat(np.arange(1, 5), 5) 
        subject = list(subject_ids) * 2  
        trial = list(np.tile(np.arange(1, 6), 4)) * 2
        df = pd.DataFrame({
            'Accuracy': accuracy,
            'Method': method,
            'Subject': subject,
            'Trial': trial
        })
        model = smf.mixedlm("Accuracy ~ Method", df, groups=df["Subject"])
        result = model.fit()
        print(result.summary())
    result_enhanced = []
    result_orginal = []
    for orginal in (1,0):
        avg_results = []
        for sub in ("S1","S2","S3","S4"):
            ps=15
            star=1
            WIN_SIZE_MS = 200
            FS=4000
            fext=16
            SEED = 42
            R = fext-1
            SUBJ = sub
            FEAT = "EMG"
            emg = load_signal(os.path.join("D:/codes/Adaptive-ICA-sEMG-TBioCAS/scripts/data"), SUBJ, task_type="gesture")
            labels = emg["Trigger"]
            s2i_map = {v: k for k, v in i2s_map_gesture.items()}
            labels1=labels.to_numpy()
            vec_map = np.vectorize(lambda s: s2i_map[s])
            labels1 = vec_map(labels1)
            # labels_fixed = labels.shift(int(round(4000)), fill_value="rest")
            emg = emg.drop(columns="Trigger")
            emg = emg.to_numpy()
            emg = emgkit.preprocessing.bandpass_filter(emg, low_cut=20, high_cut=500, fs=FS)
            num_blocks = emg.shape[0] // 800  
            processed_blocks = []
            y=[]
            st=time.time()
            if orginal==0:
                for i in range(num_blocks):
                    start = i * 800
                    end = (i + 1) * 800
                    block = emg[start:end, :]  # 获取当前块
                    processed_block = feature(emg=block,R=R,co_number=200,use_orginal_data=False,trans=True,ps=ps)
                    processed_blocks.append(processed_block)
                emg = np.transpose(np.concatenate(processed_blocks, 1))
            win_size = int(round(WIN_SIZE_MS / 1000 * FS))
            pad_size = int(round(1 * FS))
            labels_tmp=labels
            labels_tmp = labels.iloc[: emg.shape[0]]
            if isinstance(feat, int):
                feat_mapping = {
                    0: "RMS",
                    1: "TKEO",
                    2: "WL",
                    3: "VAR",
                    4: "ZC",
                    5: "SSC",
                    6: "WAMP",
                    7: "MNF",
                    8: "MDF",
                    9: "Hjorth_Activity",
                    10: "Hjorth_Mobility",
                    11: "Hjorth_Complexity",
                    12: "Spectral_Centroid",
                    13: "Spectral_Bandwidth",
                    14: "Spectral_Entropy",
                    15: "Wavelet_Energy",
                    16: "STFT_Energy",
                    17: "Spectral_Rolloff",
                    18: "petrosian_fd",
                    19: "MAV",
                    20: "ZOM",
                    21: "SOM",
                    22: "FOM",
                    23: "PS",
                    24: "LOGD",
                    25: "MAX",
                    26: "MIN",
                    27: "STD",
                    28: "SSI",
                    29: "AAC",
                    30: "DASDV",
                    31: "WLR",
                    32: "MASOD",
                    33: "ML",
                    34: "DS",
                    35: "TDPSD_F1",
                    36: "TDPSD_F2",
                    37: "TDPSD_F3",
                    38: "INVTDD_F1",
                    39: "INVTDD_F2",
                    40: "RMSS",
                    41: "PeakStress",
                    42: "MAD",
                    43: "DAMV",
                }
                FEAT = feat_mapping.get(feat, None)
                if FEAT is None:
                    raise ValueError(f"Unknown feat index: {feat}")
            else:
                FEAT = feat
            if flag == 0:
                print(FEAT)
                flag = 1
            if FEAT == "RMS":
                sig = emg
                reduce_fn = lambda x: calc_rms_all(x.T)
            elif FEAT == "TKEO":
                sig = emg
                reduce_fn = lambda x: np.mean(TKEO_fea(x), axis=0)
            elif FEAT == "WL":
                sig = emg
                reduce_fn = lambda x: calc_wl_all(x.T)
            elif FEAT == "VAR":
                sig = emg
                reduce_fn = lambda x: calc_var_all(x.T)
            elif FEAT == "ZC":
                sig = emg
                reduce_fn = lambda x: calc_zc_all(x.T)
            elif FEAT == "SSC":
                sig = emg
                reduce_fn = lambda x: calc_ssc_all(x.T)
            elif FEAT == "WAMP":
                sig = emg
                reduce_fn = lambda x: calc_wamp_all(x.T)
            elif FEAT == "MNF":
                sig = emg
                reduce_fn = lambda x: calc_mnf_all(x.T)
            elif FEAT == "MDF":
                sig = emg
                reduce_fn = lambda x: calc_mdf_all(x.T)
            elif FEAT == "PeakStress":
                sig = emg
                reduce_fn = lambda x: PeakStress(x.T)
            elif FEAT == "MAD":
                sig = emg
                reduce_fn = lambda x: calc_mad(x.T)
            elif FEAT == "DAMV":
                sig = emg
                reduce_fn = lambda x: calc_damv_all(x.T)
            elif FEAT == "Kurtosis":
                sig = emg
                reduce_fn = lambda x: calc_kurtosis_all(x.T)
            elif FEAT == "Hjorth_Activity":
                sig = emg
                reduce_fn = lambda x: calc_hjorth_activity_all(x.T)
            elif FEAT == "Hjorth_Mobility":
                sig = emg
                reduce_fn = lambda x: calc_hjorth_mobility_all(x.T)
            elif FEAT == "Hjorth_Complexity":
                sig = emg
                reduce_fn = lambda x: calc_hjorth_complexity_all(x.T)
            elif FEAT == "Spectral_Centroid":
                sig = emg
                reduce_fn = lambda x: calc_spectral_centroid_all(x.T)
            elif FEAT == "Spectral_Bandwidth":
                sig = emg
                reduce_fn = lambda x: calc_spectral_bandwidth_all(x.T)
            elif FEAT == "Spectral_Entropy":
                sig = emg
                reduce_fn = lambda x: calc_spectral_entropy_all(x.T)
            elif FEAT == "Wavelet_Energy":
                sig = emg
                reduce_fn = lambda x: calc_wavelet_energy_all(x.T)
            elif FEAT == "STFT_Energy":
                sig = emg
                reduce_fn = lambda x: calc_stft_energy_all(x.T)
            elif FEAT == "Spectral_Rolloff":
                sig = emg
                reduce_fn = lambda x: calc_spectral_rolloff_all(x.T)
            elif FEAT == "Approximate_Entropy":
                sig = emg
                reduce_fn = lambda x: calc_approx_entropy_all(x.T)
            elif FEAT == "Sample_Entropy":
                sig = emg
                reduce_fn = lambda x: calc_sample_entropy_all(x.T)
            elif FEAT == "Petrosian_FD":
                sig = emg
                reduce_fn = lambda x: calc_petrosian_fd_all(x.T)
            elif FEAT == "MAV":
                sig = emg
                reduce_fn = lambda x: calc_mav_all(x.T)
            elif FEAT == "ZOM":
                sig = emg
                reduce_fn = lambda x: calc_zom_all(x.T)
            elif FEAT == "SOM":
                sig = emg
                reduce_fn = lambda x: calc_som_all(x.T)
            elif FEAT == "FOM":
                sig = emg
                reduce_fn = lambda x: calc_fom_all(x.T)
            elif FEAT == "PS":
                sig = emg
                reduce_fn = lambda x: calc_ps_all(x.T)
            elif FEAT == "LOGD":
                sig = emg
                reduce_fn = lambda x: calc_logd_all(x.T)
            elif FEAT == "MAX":
                sig = emg
                reduce_fn = lambda x: calc_max_all(x.T)
            elif FEAT == "MIN":
                sig = emg
                reduce_fn = lambda x: calc_min_all(x.T)
            # Newly added features
            elif FEAT == "STD":
                sig = emg
                reduce_fn = lambda x: calc_std_all(x.T)
            elif FEAT == "SSI":
                sig = emg
                reduce_fn = lambda x: calc_ssi_all(x.T)
            elif FEAT == "AAC":
                sig = emg
                reduce_fn = lambda x: calc_aac_all(x.T)
            elif FEAT == "DASDV":
                sig = emg
                reduce_fn = lambda x: calc_dasdv_all(x.T)
            elif FEAT == "IRF":
                sig = emg
                reduce_fn = lambda x: calc_irf_all(x.T)
            elif FEAT == "WLR":
                sig = emg
                reduce_fn = lambda x: calc_wlr_all(x.T)
            elif FEAT == "MASOD":
                sig = emg
                reduce_fn = lambda x: calc_masod_all(x.T)
            elif FEAT == "ML":
                sig = emg
                reduce_fn = lambda x: calc_ml_all(x.T)
            elif FEAT == "DS":
                sig = emg
                reduce_fn = lambda x: calc_ds_all(x.T)
            elif FEAT == "TDPSD_F1":
                sig = emg
                reduce_fn = lambda x: calc_tdpsd_f1_all(x.T)
            elif FEAT == "TDPSD_F2":
                sig = emg
                reduce_fn = lambda x: calc_tdpsd_f2_all(x.T)
            elif FEAT == "TDPSD_F3":
                sig = emg
                reduce_fn = lambda x: calc_tdpsd_f3_all(x.T)
            elif FEAT == "INVTDD_F1":
                sig = emg
                reduce_fn = lambda x: calc_invtdd_f1_all(x.T)
            elif FEAT == "INVTDD_F2":
                sig = emg
                reduce_fn = lambda x: calc_invtdd_f2_all(x.T)
            elif FEAT == "RMSS":
                sig = emg
                reduce_fn = lambda x: calc_rmss_all(x.T)
            else:
                raise ValueError(f"Unknown feature name: {FEAT}")
            fold_results = []
            rep = [0,1,2,3,4,5,6,7,8,9]
            for val_t in [0,2,4,6,8]:
                X_train_list = []
                y_train_list = []
                X_test_list = []
                y_test_list = []
                for i_target, s_target in gesture.items():
                    chunks = segment_signal(sig, labels_tmp, s_target)
                    if len(chunks) == 0:
                        print(s_target, "empty")
                        continue
                    train_chunks = [chunks[i] for i in (rep[(val_t+0)%10],rep[(val_t+1)%10],rep[(val_t+2)%10],rep[(val_t+3)%10],rep[(val_t+4)%10],rep[(val_t+5)%10],rep[(val_t+6)%10],rep[(val_t+7)%10])]
                    for chunk in train_chunks:
                        start = pad_size
                        stop = chunk.shape[0]
                        while start + win_size < stop:
                            X_train_list.append(reduce_fn(chunk[start : start + win_size]).reshape(1, -1))
                            y_train_list.append((i_target,))
                            start += win_size-400
                    test_chunks = [chunks[i] for i in (rep[(val_t+8)%10],rep[(val_t+9)%10])]
                    for chunk in test_chunks:
                        start = pad_size
                        stop = chunk.shape[0]
                        while start + win_size < stop:
                            X_test_list.append(reduce_fn(chunk[start : start + win_size]).reshape(1, -1))
                            y_test_list.append((i_target,))
                            start += win_size-400
                X_train = np.concatenate(X_train_list,0)
                y_train = np.concatenate(y_train_list,0)
                X_test = np.concatenate(X_test_list,0)
                y_test = np.concatenate(y_test_list,0)
                # print("Shape of training data:", X_train.shape, y_train.shape)
                # print("Shape of test data:", X_test.shape, y_test.shape)
                clf = RandomForestClassifier(max_depth=None, random_state=SEED, n_estimators=100, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred_train = clf.predict(X_train)
                y_pred_test = clf.predict(X_test)
                report = classification_report(
                    y_test, y_pred_test, target_names=list(s2i_map_gesture.keys()), output_dict=True, digits=5
                )
                weighted_avg = report["weighted avg"]
                if orginal==1:
                    result_orginal.append(weighted_avg['precision'])
                else:
                    result_enhanced.append(weighted_avg['precision'])
                y_all.append(y_test)
                y_pred_all.append(y_pred_test)
                fold_results.append({
                    "fold": val_t,
                    "precision": weighted_avg["precision"],
                    "recall": weighted_avg["recall"],
                    "f1_score": weighted_avg["f1-score"]
                })
            mean_precision = np.mean([r["precision"] for r in fold_results])
            mean_recall = np.mean([r["recall"] for r in fold_results])
            mean_f1_score = np.mean([r["f1_score"] for r in fold_results])
            avg_results.append({
                "precision": mean_precision,
                "recall": mean_recall,
                "f1_score": mean_f1_score
            })
            print(
                f"\nOverall - Precision: {mean_precision:.5f}, "
                f"Recall: {mean_recall:.5f}, F1-Score: {mean_f1_score:.5f}"
            )
        avg_precision = np.mean([r["precision"] for r in avg_results])
        avg_recall = np.mean([r["recall"] for r in avg_results])
        avg_f1_score = np.mean([r["f1_score"] for r in avg_results])
