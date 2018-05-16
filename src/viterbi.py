"""
Viterbi algorithm for finding the most likely state sequence of a simple 
hidden markov model.

# https://en.wikipedia.org/wiki/Viterbi_algorithm

The Model
---------
The HMMs considered in this code are extremely simple. Every hidden state
is uniquely associated with an output symbol, but the link is "fuzzy". That is,
it has a certain probability of outputing correctly (a parameter you pick),
and if it fails, every other output is equally likely.

The other parameter is for the underlying markov chain, and is called the
metastability. The underlying markov chain has transition probabilies such that
for all states i, `T(i->i) = metastability`, and the transition probability 
to every other state j is equally likely  ( T(i->j) = (1-metastability) 
/ (n_states - 1) )

The Speed
---------
In return for all of this simplicity, you get speed. This module includes code
in multithreaded C, pure python, and an implementation that uses the python
machine learning library scikit-learn. For a problem with 100 states and 10,000
observations, the results are identitical:

c:   (-64963.11150689036, array([23, 23, 23, ..., 27, 27, 27]))
py:  (-64963.11150689036, array([23, 23, 23, ..., 27, 27, 27]))
skl: (-64963.11150689036, array([23, 23, 23, ..., 27, 27, 27]))

but the times are not.

scikit: 17.70 s
python: 26.16 s
weave:  0.482 s


This is only using two threads. If you have more cores on your machine, you can
probably do even better.
"""
import numpy as np
try:
    import scipy.weave
except:
    import weave


def viterbi(signal, metastability, p_correct):
    """
    Use the viterbi algorithm to rectify a signal for a very simple HMM.

    The underlying markov chain in the HMM is such that every i -> i transition
    has the same probability (the metastability parameter), and the transitions
    from i -> j (where j != i) are all equally likely. The emmission model
    is that each hidden state i has probablility `p_correct` of emmitting the
    signal `i`, and its probability of emmitting any other signal is equal to
    (1 - p_correct) / (n_signals - 1).
    
    This function is implemented in C using scipy.weave, and is multithreaded
    via OpenMP. To control the number of threads, you may set the environment
    variable OMP_NUM_THREADS (e.g. "export OMP_NUM_THREADS=2" in bash)

    Parameters
    ----------
    signal : np.array
    metastability : float
    p_correct : float

    Returns
    -------
    rectified : np.array
    """
    
    signal = np.array(signal)
    if signal.ndim != 1:
        raise ValueError('signal must be 1d')
    if signal.dtype != np.int:
        raise ValueError('signal must be dtype int')
    if metastability != float(metastability):
        raise TypeError('metastability must be float')
    if p_correct != float(p_correct):
        raise TypeError('p_correct must be float')
    # even something that passes the above test could be like a 32bit numpy int,
    # which is unacceptable.
    metastability = float(metastability)
    p_correct = float(p_correct)

    unique = np.unique(signal)
    if np.count_nonzero(unique - np.arange(len(unique))) != 0:
        raise ValueError('signal should contain contiguous integers from zero to its max value')
    
    n_frames = len(signal)
    n_states = len(unique)

    # declare the three arrays that we'll be using in python, so that they are
    # under python memory manaagement
    rectified = np.zeros(n_frames, dtype=np.int)    
    pointers = np.zeros((n_frames, n_states), dtype=np.int)
    V = np.ones((n_frames, n_states), dtype=np.float)
        
    scipy.weave.inline(r'''
    // these are the parameters for the log transition matrix and the log of
    // the emission probabilities
    double log_metastability = log(metastability);
    double log_off_diagnal_tprob = log((1 - metastability) / (n_states - 1));
    double log_p_correct = log(p_correct);
    double log_p_incorrect = log((1 - p_correct) / (n_states - 1));
    
    // these are used as scratch variables
    int i, t, k, argmax_of_array;
    double val, max_of_array;
    
    // fill in the first column of V
    for (i = 0; i < n_states; i++) {
        V[i] = log_p_incorrect - log(n_states);
    }
    V[signal[0]] = log_p_correct - log(n_states);
    
    
    // use dynamic programming to fill up the matrix V
    for (t = 1; t < n_frames; t++) {
        #pragma omp parallel for private(max_of_array, argmax_of_array, i, val) shared(V, pointers, t, n_states, log_p_correct, log_p_incorrect)
        for (k = 0; k < n_states; k++) {
            //printf("threads %d\n", omp_get_num_threads());
        
            //compute max_product and argmax_product
            max_of_array = -1e300;
            argmax_of_array = -1;
            for (i = 0; i < n_states; i++) {
                if (i == k) {
                    val = log_metastability + V[(t-1)*n_states + i];
                } else {
                    val = log_off_diagnal_tprob + V[(t-1)*n_states + i];
                }
                if (val > max_of_array) {
                    max_of_array = val;
                    argmax_of_array = i;
                }
            }
        
            pointers[t*n_states + k] = argmax_of_array;
            if (k == signal[t]) {
                V[t*n_states + k] = log_p_correct + max_of_array;
            } else {
                V[t*n_states + k] = log_p_incorrect + max_of_array;
            }
        }   
    }
    
    // okay now that we've filled out V, we need to work backwards
    // to reconstruct the rectified signal
    
    // set the last entry by looking at the final column of V
    max_of_array = -1e300;
    argmax_of_array = -1;
    for (i = 0; i < n_states; i++) {
        val = V[(n_frames-1)*n_states + i];
        if (val > max_of_array) {
            max_of_array = val;
            argmax_of_array = i ;
        }
    }
    rectified[n_frames - 1] = argmax_of_array;
    
    // iterate backward from te last entry towards the beginning
    // following the pointers

    for (t = n_frames - 2; t >= 0; t--) {
        rectified[t] = pointers[(t+1)*n_states + rectified[t+1]];
    }
    ''', ['signal', 'rectified', 'pointers', 'V', 'metastability', 'p_correct',
        'n_states', 'n_frames'],
        extra_link_args = ['-lgomp'], extra_compile_args = ["-O3", "-fopenmp"],
        headers=['<omp.h>'])
        
    return np.max(V[n_frames-1, :]), rectified
    
    
def _viterbi(signal, metastability, p_correct):
    """
    Use the viterbi algorithm to rectify a signal for a very simple HMM.
    
    The underlying markov chain in the HMM is such that every i -> i transition
    has the same probability (the metastability parameter), and the transitions
    from i -> j (where j != i) are all equally likely. The emmission model
    is that each hidden state i has probablility `p_correct` of emmitting the
    signal `i`, and its probability of emmitting any other signal is equal to
    (1 - p_correct) / (n_signals - 1).
    
    This is a pure python implementation
    
    Parameters
    ----------
    signal : np.array
    metastability : float
    p_correct : float
    
    Returns
    -------
    rectified : np.array
    """
    
    unique = np.unique(signal)
    n_frames = len(signal)
    n_states = len(unique)
    log_p_correct = np.log(p_correct)
    log_p_incorrect = np.log((1 - p_correct) / (n_states - 1))
    log_metastability = np.log(metastability)
    log_off_diagnal_tprob = np.log((1 - metastability) / (n_states - 1))

    # this array is awesome
    V = np.ones((n_frames, n_states))

    # declare the array of back pointers
    pointers = np.zeros((n_frames, n_states))
        
    # set the first time point. note V is holding the logs
    V[0, :]  = log_p_incorrect - np.log(n_states)
    V[0, signal[0]] = log_p_correct - np.log(n_states)
    
    def row_of_log_transition_matrix(k):
        "Get the kth row of the transition matrix"
        row =  log_off_diagnal_tprob * np.ones(n_states)
        row[k] = log_metastability
        return row
    
    # use dynamic programming to fill up the matrix V
    # http://en.wikipedia.org/wiki/Viterbi_algorithm
    for t in xrange(1, n_frames):
        for k in xrange(n_states):
            
            # do everything in log space, so this product is actually a sum
            row_log_product = row_of_log_transition_matrix(k) + V[t-1, :];
            maxval = np.max(row_log_product)
            ptr = np.argmax(row_log_product)

            pointers[t, k] = ptr            
            if k == signal[t]:
                V[t, k] = log_p_correct + maxval
            else:
                V[t, k] = log_p_incorrect + maxval
        
            
    rectified = np.zeros(n_frames, dtype=np.int)
    # set the last entry by looking at the final column of V
    rectified[n_frames - 1] = np.argmax(V[n_frames-1, :])
    # iterate backward from te last entry towards the beginning
    # following the pointers
    for t in xrange(n_frames - 2, -1, -1):
        rectified[t] = pointers[t + 1, rectified[t + 1]]

    return np.max(V[n_frames-1, :]), rectified



def viterbi_skl(signal, metastability, p_correct):
    "Use the MultinomialHMM module in scikit-learn to test the above algorithms"
    from sklearn.hmm import MultinomialHMM
    
    n_states = len(np.unique(signal))
    transmat = np.ones((n_states, n_states)) * (1-metastability) / (n_states - 1)
    emission = np.ones((n_states, n_states)) * (1-p_correct) / (n_states - 1)
    for i in range(n_states):
        transmat[i,i] = metastability
        emission[i,i] = p_correct    
    hmm = MultinomialHMM(n_components=n_states, startprob=np.ones(n_states)/n_states, transmat=transmat)
    hmm.emissionprob_ = emission

    return hmm.decode(signal)
    
    
if __name__ == '__main__':
    import time
    signal = np.random.randint(100, size=10000)
    print signal
    
    t1 = time.time()
    print 'c  ', viterbi(signal, metastability=0.99, p_correct=0.9)
    t2 = time.time()
    print 'py ', _viterbi(signal, metastability=0.99, p_correct=0.9)
    t3 = time.time()
    print 'skl', viterbi_skl(signal, metastability=0.99, p_correct=0.9)
    t4 = time.time()
    
    print t4-t3, t3-t2, t2-t1
    
                
