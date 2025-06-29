import numpy as np
import hnswlib
from numpy import linalg as LA
from collections import defaultdict
import math
from typing import Union

def cosine_sim(x,y):
    return np.dot(x,y)/LA.norm(x)/LA.norm(y)

class DataPair:
    def __init__(self, Image_path, Text, I_feature, T_feature):
        self.Image_path=Image_path
        self.Text=Text
        self.I_feature=I_feature
        self.T_feature=T_feature
        self.index = None

    def I_sim(self, point):
        return cosine_sim(self.I_feature, point.I_feature)

    def I_distance(self, points):
        if type(points).__name__=='DataPair':
            return 1.-cosine_sim(self.I_feature, points.I_feature)
        elif type(points) is list:
            return np.array([1.-cosine_sim(self.I_feature, p.I_feature) for p in points])
        else:
            raise TypeError("data should be list or DataPair!")

    def T_distance(self, points):
        if type(points).__name__=='DataPair':
            return 1.-cosine_sim(self.T_feature,points.T_feature)
        elif type(points) is list:
            return np.array([1.-cosine_sim(self.T_feature, p.T_feature) for p in points])
        else:
            raise TypeError("data should be list or DataPair!")


class DataPoint:
    def __init__(self, Image_path, I_feature, label, confidence=0.9):
        self.Image_path=Image_path
        self.label=label
        self.I_feature=I_feature
        self.index = None
        self.confidence = confidence

    def I_sim(self, point):
        return cosine_sim(self.I_feature, point.I_feature)
    
    def I_distance(self, points):
        if type(points).__name__=='DataPoint':
            return 1.-cosine_sim(self.I_feature, points.I_feature)
        elif type(points) is list:
            return np.array([1.-cosine_sim(self.I_feature, p.I_feature) for p in points])
        else:
            raise TypeError("data should be list or DataPoint!")
    

class Multimodal_index:
    # New incoming data should have comparable cleaness as bootstraping set.
    # Current version handle million scale on single machine; 
    # For larger scale, consider using hyperplane (vector space separation) 
    # to further distribute samples across machines;
    # and the HNSW can be further modified or replaced for better concurrency.

    def __init__(self, initial_points=None, n=3000000, keep_seed=False, submodular_k=4):
        self.clusters = None
        self.target_n = n
        self.current_n = 0
        self.keep_seed = keep_seed
        self.dim = 256
        self.submodular_k = submodular_k

        if initial_points: 
            self.data = initial_points.copy()
            self.current_n = len(initial_points)
            if keep_seed:
                self.seeded_num = len(initial_points)
            self.dim = len(initial_points[0].I_feature)
        else:
            self.data = []
        self.submodular_gain = [(1,1)]*len(self.data)

        # initialize HNSW index
        self.I_knn_graph = hnswlib.Index(space='cosine', dim=self.dim)
        self.I_knn_graph.init_index(max_elements=n, ef_construction=100, M=48, allow_replace_deleted = False)
        self.T_knn_graph = hnswlib.Index(space='cosine', dim=self.dim)
        self.T_knn_graph.init_index(max_elements=n, ef_construction=100, M=48, allow_replace_deleted = False)
        self.precluster(initial_points)

        self.I_knn_graph.set_ef(32)
        self.T_knn_graph.set_ef(32)
        self.min_align = 0.4


    def precluster(self, initial_points):
    # Starting from some initial points (the cleaner the better) to do online selection
        if initial_points is None or initial_points==[]: return
        for idx,data in enumerate(self.data):
            data.index = idx

        for idx,data in enumerate(self.data):
            self.submodular_gain[idx] = self.submodular_func(data, True)
            self.I_knn_graph.add_items(data.I_feature, idx)
            self.T_knn_graph.add_items(data.T_feature, idx)
            

    def submodular_func(self, data, skip_one=False):
        if self.I_knn_graph.get_current_count()==0:
            return (1.,1.)
        k = min(self.I_knn_graph.get_current_count(), self.submodular_k)
        
        I_near_labels, I_near_distances = self.k_nearest_neighbour_I(data, k)
        T_near_labels, T_near_distances = self.k_nearest_neighbour_T(data, k)
        return (np.mean(I_near_distances),np.mean(T_near_distances))

    def align_score(self,data):
        if type(data).__name__=='DataPair':
            return cosine_sim(data.I_feature,data.T_feature)
        elif type(data) is list:
            return [self.align_score(x) for x in data]
        else:
            raise TypeError("data should be list or DataPair!")

    def k_nearest_neighbour_I(self, data, k):
        I_near_labels, I_near_distances = self.I_knn_graph.knn_query(data.I_feature, k)
        return I_near_labels, I_near_distances

    def k_nearest_neighbour_T(self,data, k):
        T_near_labels, T_near_distances = self.T_knn_graph.knn_query(data.T_feature, k)
        return T_near_labels, T_near_distances

    def I_to_T_k_nearest(self, data, k):
        T_near_labels, T_near_distances = self.T_knn_graph.knn_query(data.I_feature, k)
        return T_near_labels, T_near_distances

    def T_to_I_k_nearest(self, data, k):
        I_near_labels, I_near_distances = self.I_knn_graph.knn_query(data.T_feature, k)
        return I_near_labels, I_near_distances

    def add_item(self, data):
        data.index = self.current_n
        self.data.append(data)
        self.I_knn_graph.add_items(data.I_feature, self.current_n)
        self.T_knn_graph.add_items(data.T_feature, self.current_n)
        self.current_n+=1

    def replace_item(self, data, index):
        # Not used in current work but provide for future extension on replacing samples
        # replace data_old at index with data
        data_to_rep = self.data[index]
        n_index = data_to_rep.index
        data.index = index
        self.I_knn_graph.mark_deleted(n_index)
        self.T_knn_graph.mark_deleted(n_index)
        self.I_knn_graph.add_items(data.I_feature, index, replace_deleted = True)
        self.T_knn_graph.add_items(data.T_feature, index, replace_deleted = True)
        self.data[index] = data

    def process_item(self, data: DataPair, recaptioner = None):
        # find near clusters
        # go into nearest clusters to search near neighbour
        # calculate corresponding threshold to decide if try to add or not
        align_score = self.align_score(data)
        if recaptioner and data.Image_path in recaptioner:
            text = recaptioner[data.Image_path]['caption']
            recap_T_feature = recaptioner[data.Image_path]['text_feature']
            recap_align_score = cosine_sim(data.I_feature,recap_T_feature)
            if align_score<0.4 and recap_align_score>=0.4:
                align_score = recap_align_score
                data.Text = text
                data.T_feature = recap_T_feature

        if align_score<self.min_align:
            return

        gain = self.submodular_func(data)

        self.add_item(data)
        self.submodular_gain.append(gain)

    def final_gains(self):
        return self.submodular_gain
    
class Singlemodal_index:
    # New incoming data should have comparable cleaness as bootstraping set.
    # Current version handle million scale on single machine; 
    # For larger scale, consider using hyperplane (vector space separation) 
    # to further distribute samples across machines;
    # and the HNSW can be further modified or replaced for better concurrency.

    def __init__(self, initial_points=None, dim=128, n=3000000, keep_seed=False, submodular_k=8, d_threshold=0.15, num_classes=1000):
        self.clusters = None
        self.target_n = n
        self.current_n = 0
        self.keep_seed = keep_seed
        self.dim = dim
        self.submodular_k = submodular_k

        if initial_points: 
            self.data = initial_points.copy()
            self.current_n = len(initial_points)
            if keep_seed:
                self.seeded_num = len(initial_points)
            self.dim = len(initial_points[0].I_feature)
        else:
            self.data = []
        self.submodular_gain = [1]*len(self.data)

        # initialize HNSW index
        self.I_knn_graph = hnswlib.Index(space='cosine', dim=self.dim)
        self.I_knn_graph.init_index(max_elements=n, ef_construction=100, M=48, allow_replace_deleted = False)
        self.precluster(initial_points)

        self.I_knn_graph.set_ef(32)

        self.distance_threshold=d_threshold
        self.num_classes = num_classes


    def precluster(self, initial_points):
    # Starting from some initial points (the cleaner the better) to do online selection
        if initial_points is None or initial_points==[]: return
        for idx,data in enumerate(self.data):
            data.index = idx

        for idx,data in enumerate(self.data):
            self.submodular_gain[idx] = self.submodular_func(data, True)
            self.I_knn_graph.add_items(data.I_feature, idx)
            

    def knn_classifier(self, classes, confidences, sim, normalize=True):
        # noted that skip_one is not implemented for this function yet. Only for incoming data or internel function.
        dic = defaultdict(float)
        for i in range(len(classes)):
            dic[classes[i]]+=confidences[i]*sim[i]
        total_weight = sum(dic.values())
        max_c, max_conf = None, 0
        for c, conf in dic.items():
            if conf>max_conf:
                max_c = c
                max_conf = conf
        if normalize:
            return max_c, max_conf/total_weight if total_weight>0 else max_conf
        else:
            return max_c, max_conf, total_weight

    def sum_neighbour(self, classes, confidences, sim, normalize=True):
        dic = defaultdict(float)
        for i in range(len(classes)):
            dic[classes[i]]+=confidences[i]*sim[i]
        total_weight = sum(dic.values())
        if normalize and total_weight>0:
            for c in dic:
                dic[c]/=total_weight
        return dic

    def knn_pred(self, data, k, skip_one=False, normalize=True):
        if self.I_knn_graph.get_current_count()==0:
            return 1.
        k = min(self.I_knn_graph.get_current_count(), self.submodular_k)
        
        I_near_labels, I_near_distances = self.k_nearest_neighbour_I(data, k, skip_one=skip_one)

        ###New Calculation: how the new sample reduce the entropy###
        selected_ids = I_near_labels[I_near_distances<=self.distance_threshold]
        sim = 1-I_near_distances[I_near_distances<=self.distance_threshold]
        classes = np.array([self.data[idx].label for idx in selected_ids])
        confidences = np.array([self.data[idx].confidence for idx in selected_ids])
        if len(classes):
            preds = self.knn_classifier(classes, confidences, sim, normalize=normalize)
        else:
            preds = (0,0)
        return preds

    def knn_pred_dic(self, data, k=8, skip_one=False, normalize=True):
        if self.I_knn_graph.get_current_count()==0:
            return 1.
        k = min(self.I_knn_graph.get_current_count(), self.submodular_k)
        
        I_near_labels, I_near_distances = self.k_nearest_neighbour_I(data, k, skip_one=skip_one)

        ###New Calculation: how the new sample reduce the entropy###
        selected_ids = I_near_labels[I_near_distances<=self.distance_threshold]
        sim = 1-I_near_distances[I_near_distances<=self.distance_threshold]
        classes = np.array([self.data[idx].label for idx in selected_ids])
        confidences = np.array([self.data[idx].confidence for idx in selected_ids])
        if len(classes):
            preds = self.sum_neighbour(classes, confidences, sim, normalize=normalize)
        else:
            preds = {}
        return preds

    # naive version of implementation, how the new data benefit a dataset's view on given point
    def submodular_func(self, data, skip_one=False):
        # a model's view is also possible
        if self.I_knn_graph.get_current_count()==0:
            return 1.
        k = min(self.I_knn_graph.get_current_count(), self.submodular_k)
        
        I_near_labels, I_near_distances = self.k_nearest_neighbour_I(data, k)

        if skip_one:
            id = np.argmin(I_near_distances)
            I_near_labels=np.delete(I_near_labels,id,axis=0)
            I_near_distances=np.delete(I_near_distances,id,axis=0)

        ###New Calculation: how the new sample reduce the entropy###
        selected_ids = I_near_labels[I_near_distances<=self.distance_threshold]
        sim = 1-I_near_distances[I_near_distances<=self.distance_threshold]
        # if max(sim)>0.99: return 0
        classes = np.array([self.data[idx].label for idx in selected_ids])
        confidences = np.array([self.data[idx].confidence for idx in selected_ids])
        if len(classes):
            preds = self.sum_neighbour(classes, confidences, sim)
            # linear_combined_pred = self.sum_neighbour()
            original_entropy = sum([x*math.log(x) for x in preds.values()])
        else:
            original_entropy = -math.log(1/self.num_classes)
        c = data.confidence
        assert 0<=c<=1
        if c==1.:
            entropy = 0.
        elif c==0:
            entropy = 1.
        else:
            entropy = -c*math.log(c)-(1-c)*math.log(1-c)
        
        gain = original_entropy-entropy  
        # add neighbour effect later
        return gain
        
    def safe_entropy_calculation(self, preds, num_classes=None):
        num_classes = self.num_classes if num_classes is None else num_classes
        num_classes = 10 if num_classes is None else num_classes
        if preds:
            original_entropy = -sum([0 if (x==0 or x==1) else x*math.log(x) for x in preds.values()])
        else:
            original_entropy = -math.log(1/num_classes)
        return original_entropy

    def entropy_from_dataset_view(self, data, k=8, skip_one=False, num_classes=None):
        preds = self.knn_pred_dic(data,k,skip_one=skip_one)
        return self.safe_entropy_calculation(preds,num_class=num_classes)
        
    def k_nearest_neighbour_I(self, data, k, skip_one=False):
        k = min(self.I_knn_graph.get_current_count(), k+int(skip_one))

        if isinstance(data, DataPoint):
            I_near_labels, I_near_distances = self.I_knn_graph.knn_query(data.I_feature, k)
        elif isinstance(data, np.ndarray):
            I_near_labels, I_near_distances = self.I_knn_graph.knn_query(data, k)
        else:
            raise ValueError(f'{data} with type {type(data)} not compatible for ANN search!')

        if len(I_near_labels.shape) > 1 and I_near_labels.shape[0]==1:
            I_near_labels = I_near_labels[0]
            I_near_distances = I_near_distances[0]
            
        if skip_one:
            if isinstance(data, DataPoint):
                selected = I_near_labels!=data.index
                I_near_labels=I_near_labels[selected]
                I_near_distances=I_near_distances[selected]
            else:
                id = np.argmin(I_near_distances)
                I_near_labels=np.delete(I_near_labels,id,axis=0)
                I_near_distances=np.delete(I_near_distances,id,axis=0)
        
        return I_near_labels, I_near_distances

    def add_item(self, data):
        data.index = self.current_n
        self.data.append(data)
        self.I_knn_graph.add_items(data.I_feature, self.current_n)
        self.current_n+=1

    def replace_item(self, data, index):
        # Not used in current work but provide for future extension on replacing samples
        data_to_rep = self.data[index]
        n_index = data_to_rep.index
        data.index = index
        self.I_knn_graph.mark_deleted(n_index)
        self.I_knn_graph.add_items(data.I_feature, index, replace_deleted = True)
        self.data[index] = data

    def process_item(self, data: DataPoint):
        gain = self.submodular_func(data)
        if gain>0:
            self.add_item(data)
            self.submodular_gain.append(gain)
        else:
            return

    def update_static_gain(self):
        for idx,data in enumerate(self.data):
            self.submodular_gain[idx] = self.submodular_func(data,skip_one=True)

    def final_gains(self):
        return self.submodular_gain

    def get_sim_list(self, k=None):
        if not k:
            k = self.submodular_k
        sim_list = [[] for _ in range(len(self.data))]
        for idx,data in enumerate(self.data):
            I_near_labels, I_near_distances = self.k_nearest_neighbour_I(data, k, skip_one=True)
            sim_list[data.index].append(I_near_labels[I_near_distances<self.distance_threshold].tolist())
            sim_list[data.index].append(np.clip(1-I_near_distances[I_near_distances<self.distance_threshold],-1,1).tolist())
        return sim_list
    
    def get_sim_list_for_cscmartix(self, k=None):
        if not k:
            k = self.submodular_k
        row=[]
        col=[]
        return_data=[]
        for idx,data in enumerate(self.data):
            I_near_labels, I_near_distances = self.k_nearest_neighbour_I(data, k+1, skip_one=False)
            mask = np.logical_and(I_near_distances<self.distance_threshold, I_near_labels!=idx)
            l = I_near_labels[mask].tolist()
            col.extend(l)
            row.extend([data.index]*len(l))
            return_data.extend(np.clip(1-I_near_distances[mask],-1,1).tolist())
        return return_data,row,col
    
    def _gain_for_sigle_data_reannotate(self, data, relabel_confidence=0.9, skip_one=True):
        original_entropy = self.entropy_from_dataset_view(self, data, skip_one=skip_one)
        if self.num_classes:
            estimated_new_entropy = -(math.log(relabel_confidence)+(1-relabel_confidence)*math.log(1/(self.num_classes-1)))
        else:
            estimated_new_entropy = -(math.log(relabel_confidence)+(1-relabel_confidence)*math.log(0.5))
        return original_entropy-estimated_new_entropy
    

def confidence_convergence(pred1: Union[dict, list, tuple], pred2: Union[dict, list, tuple], conf_decay=False) -> dict:
    # When predictions agree with each other, it suggests increased probability
    # x,y be the probability of two independent predictions, if x,y >0.5, z = x*(1-y)/(x*(1-y)+(1-x)*y)
    # is the new probability for confidence lower bound when prediction px=py; if any one smaller than 0.5, use the max one.
    # when px!=py, let x>y, z = x*(1-y)/(x*(1-y)+(1-x)*y), which means the probability of x being true and y being wrong.
    if isinstance(pred1, dict):
        p1,c1 = pred1['class'], pred1['confidence']
    else:
        p1,c1 = pred1

    if isinstance(pred2, dict):
        p2,c2 = pred2['class'], pred2['confidence']
    else:
        p2,c2 = pred2
    
    if p1 == p2:
        x = max(c1,c2)
        y = min(c1,c2)
        if y>0.5:
            c = x*y/(x*y+(1-x)*(1-y))
        else:
            c = x
        p = p1
    else:
        p = [p1,p2][np.argmax([c1,c2])]
        if conf_decay:
            x = max(c1,c2)
            y = min(c1,c2)
            if y>0.5:
                c = x*(1-y)/(x*(1-y)+(1-x)*y)
            else:
                c = x
        else:
            c = max(c1,c2)
    return p,c

def confidence_convergence_variant1(pred1: Union[dict, list, tuple], pred2: Union[dict, list, tuple], conf_decay=False) -> dict:
    # When predictions agree with each other, it suggests increased probability
    # x,y be the probability of two independent predictions, if x,y >0.5, z = x*(1-y)/(x*(1-y)+(1-x)*y)
    # is the new probability for confidence lower bound when prediction px=py; if any one smaller than 0.5, use the max one.
    # when px!=py, let x>y, z = x*(1-y)/(x*(1-y)+(1-x)*y), which means the probability of x being true and y being wrong.
    # ablation: for prediction, using original gives higher acc
    if isinstance(pred1, dict):
        p1,c1 = pred1['class'], pred1['confidence']
    else:
        p1,c1 = pred1

    if isinstance(pred2, dict):
        p2,c2 = pred2['class'], pred2['confidence']
    else:
        p2,c2 = pred2
    
    if p1 == p2:
        x = max(c1,c2)
        y = min(c1,c2)
        if y>0.5:
            c = x*y/(x*y+(1-x)*(1-y))
        else:
            c = x
        p = p1
    else:
        p = p1
        if conf_decay:
            x,y = c1,c2
            if y>0.5:
                c = x*(1-y)/(x*(1-y)+(1-x)*y)
            else:
                c = x
        else:
            c = c1
    return p,c

def get_max_prediction(dic, normalize=True):
    max_c, max_conf = None,0
    for c, conf in dic.items():
        if conf>max_conf:
            max_c = c
            max_conf = conf
    if normalize:
        max_conf=max_conf/sum(dic.values()) if max_conf>0 else 0
    return max_c, max_conf
