import numpy as np
from scipy.special import gammaln
import re
import matplotlib.pyplot as plt


class LDACGS:
    """Do LDA with Gibbs Sampling."""
    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """Initialize system parameters."""
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def build_corpus(self, filename, stopwords_file=None):
        """Read the given filename and build the vocabulary."""
        with open(filename, 'r') as infile:
            doclines = [re.sub(r'[^\w ]', '', line.lower()).split(' ') for line in infile]

        n_docs = len(doclines)
        self.vocab = list({v for doc in doclines for v in doc})
        if stopwords_file:
            with open(stopwords_file, 'r') as stopfile:
                stops = stopfile.read().split()

            self.vocab = [x for x in self.vocab if x not in stops]
        self.vocab.sort()

        self.documents = []
        for i in range(n_docs):
            self.documents.append({})
            for j in range(len(doclines[i])):
                if doclines[i][j] in self.vocab:
                    self.documents[i][j] = self.vocab.index(doclines[i][j])

    def initialize(self):
        """Initialize the three count matrices."""
        self.n_words = len(self.vocab)
        self.n_docs = len(self.documents)
        # Initialize the three count matrices.
        # The (i,j) entry of self.nmz is the number of words in document i assigned to topic j.
        self.nmz = np.zeros((self.n_docs, self.n_topics))
        # The (i,j) entry of self.nzw is the number of times term j is assigned to topic i.
        self.nzw = np.zeros((self.n_topics, self.n_words))
        # The (i)-th entry is the number of times topic i is assigned in the corpus.
        self.nz = np.zeros(self.n_topics)
        # Initialize the topic assignment dictionary.
        self.topics = {}  # key-value pairs of form (m,i):z

        for m in range(self.n_docs):
            for i in self.documents[m]:
                # Problem 3:
                # Get random topic assignment, i.e. z = ...
                z = np.random.choice(self.n_topics)
                # Retrieve vocab index for i-th word in document m.
                w = self.documents[m][i]
                # Increment count matrices
                self.nmz[m][z] += 1
                self.nzw[z][w] += 1
                self.nz[z] += 1
                # Store topic assignment, i.e. self.topics[(m,i)]=z
                self.topics[(m, i)] = z

    def sample(self, filename, burnin=100, sample_rate=10, n_samples=10, stopwords=None):
        self.build_corpus(filename, stopwords)

        self.initialize()
        self.total_nzw = np.zeros((self.n_topics, self.n_words))
        self.total_nmz = np.zeros((self.n_docs, self.n_topics))
        self.logprobs = np.zeros(burnin + sample_rate * n_samples)

        # Problem 5:
        for i in range(burnin):
            # Sweep and store log likelihood.
            self._sweep()
            self.logprobs[i] = self._loglikelihood()
        for i in range(n_samples * sample_rate):
            # Sweep and store log likelihood
            self._sweep()
            self.logprobs[burnin + i] = self._loglikelihood()

            if not i % sample_rate:
                # accumulate counts
                self.total_nmz += self.nmz
                self.total_nzw += self.nzw

    def phi(self):
        phi = self.total_nzw + self.beta
        self._phi = phi / np.sum(phi, axis=1)[:, np.newaxis]

    def theta(self):
        theta = self.total_nmz + self.alpha
        self._theta = theta / np.sum(theta, axis=1)[:, np.newaxis]

    def topterms(self, n_terms=10):
        self.phi()
        self.theta()
        vec = np.atleast_2d(np.arange(0, self.n_words))
        topics = []
        for k in range(self.n_topics):
            probs = np.atleast_2d(self._phi[k, :])
            mat = np.append(probs, vec, 0)
            sind = np.array([mat[:, i] for i in np.argsort(mat[0])]).T
            topics.append([self.vocab[int(sind[1, self.n_words - 1 - i])] for i in range(n_terms)])
        return topics

    def toplines(self, n_lines=5):
        lines = np.zeros((self.n_topics, n_lines))
        for i in range(self.n_topics):
            args = np.argsort(self._theta[:, i]).tolist()
            args.reverse()
            lines[i, :] = np.array(args)[0:n_lines] + 1
        return lines

    def _remove_stopwords(self, stopwords):
        return [x for x in self.vocab if x not in stopwords]

    def _conditional(self, m, w):
        dist = (self.nmz[m, :] + self.alpha) * (self.nzw[:, w] + self.beta) / (self.nz + self.beta * self.n_words)
        return dist / np.sum(dist)

    def _sweep(self):
        for m in range(self.n_docs):
            for i in self.documents[m]:
                # Problem 4:
                # Retrieve vocab index for i-th word in document m.
                w = self.documents[m][i]
                # Retrieve topic assignment for i-th word in document m.
                z = self.topics[(m, i)]
                # Decrement count matrices.
                self.nmz[m][z] -= 1
                self.nzw[z][w] -= 1
                self.nz[z] -= 1
                # Get conditional distribution.
                dist = self._conditional(m, w)
                # Sample new topic assignment.
                new_z = np.random.choice(self.n_topics, p=dist)
                # Increment count matrices.
                self.nmz[m][new_z] += 1
                self.nzw[new_z][w] += 1
                self.nz[new_z] += 1
                # Store new topic assignment.
                self.topics[(m, i)] = new_z

    def _loglikelihood(self):
        lik = 0

        for z in range(self.n_topics):
            lik += np.sum(gammaln(self.nzw[z, :] + self.beta)) - gammaln(np.sum(self.nzw[z, :] + self.beta))
            lik -= self.n_words * gammaln(self.beta) - gammaln(self.n_words * self.beta)
            for m in range(self.n_docs):
                lik += np.sum(gammaln(self.nmz[m, :] + self.alpha)) - gammaln(np.sum(self.nmz[m, :] + self.alpha))
                lik -= self.n_topics * gammaln(self.alpha) - gammaln(self.n_topics * self.alpha)
        return lik


def plot_logprobs(logprobs):
    # Problem 6:
    plt.plot(logprobs)
    plt.show()


def main():
    fpath = "./data/reagan.txt"
    swpath = "./data/stopwords_en.txt"

    lda = LDACGS(10)

    lda.sample(fpath, stopwords=swpath, burnin=20)
    for i, topic in enumerate(lda.topterms()):
        print(i, topic)
    plot_logprobs(lda.logprobs)


if __name__ == '__main__':
    np.random.seed(1)
    main()
