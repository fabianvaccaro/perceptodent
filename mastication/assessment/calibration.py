import numpy as np
from mastication.capture import sampling
from mastication.assessment import instrumentation
import glob
from multiprocessing import Process, Pool
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def process_sample_job(fileinfo):
    sampleimage = sampling.FromImage(fileinfo['filename'], fileinfo['t_value'])
    mfc = instrumentation.cmfc(sampleimage)
    d = {'t_value': fileinfo['t_value'], 'mfc': mfc}
    return d

class ProcessSampleWorker(Process):
    def __init__(self, filename, t_value, dataset_queue):
        assert isinstance(dataset_queue, Queue)
        self.dataset_queue = dataset_queue
        self.filename = filename
        self.t_value = t_value
        Process.__init__(self)

    def run(self):
        sampleimage = sampling.FromImage(self.filename, self.t_value)
        mfc = instrumentation.cmfc(sampleimage)
        self.dataset_queue.put_nowait(mfc)

def preproc(folderlist):
    num_images = 0
    for f in folderlist:
        for filename in glob.glob('mastication/examples/'+f+'/*.tif'):
            num_images += 1
    return num_images


def calibrate_samples():
    folderlist = ['5', '10', '15', '20']
    # folderlist = ['5', '10']
    if __name__ == "mastication.assessment.calibration":
        iters_per_h = 10
        k0 = 40
        kmax = 120
        pool = Pool(8)
        fileinfo = []
        for f in folderlist:
            for filename in glob.glob('mastication/examples/'+f+'/*.tif'):
                fileinfo.append({'filename': filename, 't_value': int(f)})
        results = pool.map(process_sample_job, fileinfo)
        mfc_matrix = np.zeros((len(results), 121))
        targets = np.zeros(len(results))
        for y in range(0, len(results)):
            mfc = results[y]
            targets[y] = mfc['t_value']
            for x in range(0, 121):
                mfc_matrix[y, x] = mfc['mfc'][x]
        pca = PCA(n_components=3)
        pca.fit(mfc_matrix)
        dataset = pca.transform(mfc_matrix)
        # Binary cascade
        cascade = {}
        num_clases = len(folderlist)
        for target_class in folderlist:
            tgc = int(target_class) # Target class expressed as an int
            nc = [int(t) for t in folderlist if t != target_class] # Null classes
            b_targets = np.zeros(targets.shape[0])
            for tt in range(0, targets.shape[0]):
                if targets[tt] == tgc:
                    b_targets[tt] = 1
            classifiers = []
            for h in range(k0, kmax):
                for i in range(0, iters_per_h):
                    clf = MLPClassifier(hidden_layer_sizes=(h,))
                    clf.fit(dataset, b_targets)
                    predd = clf.predict(dataset)
                    mcc = matthews_corrcoef(b_targets, predd)
                    classifiers.append({'clf': clf, 'mcc': mcc})
            best_mcc = 0
            best_clf = None
            for c in classifiers:
                if abs(c['mcc'] > best_mcc):
                    best_mcc = c['mcc']
                    best_clf = c['clf']
            assert isinstance(best_clf, MLPClassifier)
            cascade[target_class] = best_clf
            print('MCC for ' + target_class + ': ' + str(best_mcc))

        print('Predictions')
        for lad in cascade:
            assert isinstance(cascade[lad], MLPClassifier)
            print(cascade[lad].predict(dataset))

        print('Calibration completed')

        return 0
    else:
        return 0

def test_calibration():
    folderlist = ['5', '10', '15', '20']
    if __name__ == "mastication.assessment.calibration":
        iters_per_h = 10
        k0 = 40
        kmax = 120
        pool = Pool(8)
        fileinfo = []
        for f in folderlist:
            for filename in glob.glob('mastication/examples/' + f + '/*.tif'):
                fileinfo.append({'filename': filename, 't_value': int(f)})
        for finfo in fileinfo:
            print(process_sample_job(finfo))
        return 0