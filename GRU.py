"""
This is a batched GRU forward and backward pass
Obviously forked from Andrej Karpathy's Python LSTM
    -did not touch Karpathy's excellent gradient checking algos
Structure of GRU from Colah's diagram
 http://colah.github.io/posts/2015-08-Understanding-LSTMs/
"""
import numpy as np


class GRU:
    @staticmethod
    def init(input_size, hidden_size):  # , fancy_forget_bias_init=3
        """
        Initialize parameters of the GRU (both weights and biases in one matrix)
        """
        # +1 for the biases, which will be the first row of WGRU
        WGRU = np.random.randn(input_size + hidden_size + 1, 3 * hidden_size) / np.sqrt(input_size + hidden_size)
        # WGRU[0, :] = 0  # initialize biases to zero
        # if fancy_forget_bias_init != 0:
            # WGRU[0, :hidden_size] = -fancy_forget_bias_init
        return WGRU

    @staticmethod
    def forward(X, WGRU, h0=None):
        """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        Copied Karpathy's LSTM design as much as possible for GRU
        Kept Weights in one big matrix WGRU
        Kept Gates in one big matrix ZRC before non-linearity
            and matrix ZRC_n after non-linearity
        """
        n, b, input_size = X.shape

        d = WGRU.shape[1] / 3  # hidden size

        # print "n = length of sequence, " , n
        # print "b = batch size, ", b
        # print "d = hidden size, ", d
        # print "input_size ", input_size

        if h0 is None: h0 = np.zeros((b, d))

        # Perform the GRU forward pass with X as the input
        xphpb = WGRU.shape[0]  # x plus h plus bias
        X_and_H_in = np.zeros((n, b, xphpb))  # input [1, xt, ht-1] to each tick of the GRU
        Hout = np.zeros((n, b, d))
        # hidden representation of the GRU (gated cell content)
        ZRC = np.zeros((n, b, d * 3))  # update z, reset r, and candidate c (ZRC)
        ZRC_n = np.zeros((n, b, d * 3))  # after nonlinearity (sigma, sigma, and tanh, respectively)
        X_and_H_reset = np.zeros((b, xphpb))  # after reset gate multiply

        for t in xrange(n):
            # concat [x,h] as input to the GRU
            prevh = Hout[t - 1] if t > 0 else h0
            X_and_H_in[t, :, 0] = 1  # bias
            X_and_H_in[t, :, 1:input_size + 1] = X[t]
            X_and_H_in[t, :, input_size + 1:] = prevh

            # compute update z and reset r gate activations. dots: (most work is this line)
            ZRC[t, :, :2*d] = X_and_H_in[t].dot(WGRU[:, :2*d]) # [2x8] = [2,11]x[11,8]
            # non-linearities
            ZRC_n[t, :, :2*d] = 1.0 / (1.0 + np.exp(-ZRC[t, :, :2*d]))  # sigmoids; these are the gates

            # temp memory content
            X_and_H_reset[:, 0] = 1  # bias
            X_and_H_reset[:, 1:input_size + 1] = X[t]
            X_and_H_reset[:, input_size + 1:] =  ZRC_n[t, :, 1*d : 2*d] * prevh #  r * h_t-1

            # summation below tanh
            ZRC[t, :, 2*d:] = X_and_H_reset.dot(WGRU[:, 2*d:])
            # non-linearity
            ZRC_n[t, :, 2*d:] = np.tanh(ZRC[t, :, 2*d:])  # tanh

            # compute the cell activation
            # prevc = C[t-1] if t > 0 else c0
            # C[t] = ZRC_n[t,:,:d] * ZRC_n[t,:,3*d:] + ZRC_n[t,:,d:2*d] * prevc

            Hout[t] = (1 - ZRC_n[t,:,:d]) * prevh + ZRC_n[t,:,:d] * ZRC_n[t, :, 2*d:] #  (1-zt)*h_t-1 + zt*C

        cache = {}
        cache['WGRU'] = WGRU
        cache['Hout'] = Hout
        cache['ZRC_n'] = ZRC_n
        cache['ZRC'] = ZRC
        cache['X_and_H_in'] = X_and_H_in
        cache['h0'] = h0


        return Hout, Hout[t], cache

    @staticmethod
    def backward(dh_out_parameter_into_backward_function, cache, dhn=None):

        WGRU = cache['WGRU']
        Hout = cache['Hout']
        ZRC_n = cache['ZRC_n']
        ZRC = cache['ZRC']

        X_and_H_in = cache['X_and_H_in']
        h0 = cache['h0']
        n, b, d = Hout.shape
        if h0 is None: h0 = np.zeros((b, d))
        input_size = WGRU.shape[0] - d - 1  # -1 due to bias

        # backprop the GRU
        dZRC = np.zeros(ZRC.shape)
        dZRC_n = np.zeros(ZRC_n.shape)  # non-linear Z, R, and C after sigma, sigma and tanh respectively
        dWGRU = np.zeros(WGRU.shape)
        dX_and_H_in = np.zeros(X_and_H_in.shape)
        dX = np.zeros((n, b, input_size))
        dh0 = np.zeros((b, d))
        # xphpb = WGRU.shape[0]  # x plus h plus bias
        # X_and_H_reset = np.zeros((b, xphpb))  # after nonlinearity

        dh_out = dh_out_parameter_into_backward_function.copy()  # make a copy so we don't have any funny side effects
        if dhn is not None: dh_out[n - 1] += dhn.copy()

        # Loop Time Backwards
        for t in reversed(xrange(n)):

            # temp memory content
            prevh = Hout[t - 1] if t > 0 else h0
            X_and_H_reset = X_and_H_in[t].copy()  # bias
            X_and_H_reset[:, input_size + 1:] = ZRC_n[t, :, 1*d : 2*d] * prevh #  R_n * h_t-1

            # above tanh
            dZRC_n[t, :, 2*d:] = ZRC_n[t, :, :d] * dh_out[t]  # dc_n = z_n * dht

            #  below tanh
            dZRC[t, :, 2*d:] = (1 - ZRC_n[t,:,2*d:]**2) * dZRC_n[t, :, 2*d:]  # dC = (1-tanh(C)^2)*dc_n

            # summation below tanh
            # dWc-xh = XH_reset.transpose * dC ## need += to accumulate over t loop
            dWGRU[:, 2*d:] += np.dot(X_and_H_reset.transpose(), dZRC[t, :, 2*d:])
            dx_and_dh_reset = dZRC[t, :, 2*d:].dot((WGRU[:, 2*d:]).transpose())

            # First component of two to add into dX
            dX_and_H_in[t,:, 1:input_size + 1] = dx_and_dh_reset[:, 1:input_size + 1]

            #  near multiply unit above reset sigma
            #  previous h times reset gate
            dZRC_n[t, :, 1*d:2*d] = prevh * dx_and_dh_reset[:, input_size + 1:]  # dr = h(t-1) * dh_reset
            #First Component for H_in
            dX_and_H_in[t,:, input_size + 1:] = ZRC_n[t,:,1*d:2*d] * dx_and_dh_reset[:, input_size + 1:]

            # near multiply gate above "1-z"
            one_minus_z = 1 - ZRC_n[t,:,:1*d] # 1-z
            d_one_minus_z = prevh * dh_out[t]
            # second of three dH components
            dX_and_H_in[t,:, input_size + 1:] += one_minus_z * dh_out[t]

            # above sigma (z)
            # first of two dZ components
            dZRC_n[t, :, :d] = ZRC_n[t, :, 2*d:] * dh_out[t] # dz2 = tanh(C) * dh ## from multiply above tanh
            # second of two dZ components
            dZRC_n[t, :, :d] += -1 * d_one_minus_z

            # below two sigma gates
            dZRC[t, :, :2*d] = ZRC_n[t, :, :2*d]*(1-ZRC_n[t, :, :2*d])*dZRC_n[t, :, :2*d]
            # sigma(Z)*(1-sigma(Z))*dZ_n   AND # sigma(R)*(1-sigma(R))*dR_n
            # summation below sigma gates
            dWGRU[:, :2*d] += np.dot(X_and_H_in[t].transpose(), dZRC[t, :, :2*d])
            # 2nd of 2 dX and 3rd of 3 dH
            dX_and_H_in[t] += dZRC[t, :, :2*d].dot((WGRU[:, :2*d]).transpose())

           
            dX[t] = dX_and_H_in[t, :, 1:input_size + 1]
            if t > 0:
                dh_out[t - 1, :] += dX_and_H_in[t, :, input_size + 1:]
            else:
                dh0 += dX_and_H_in[t, :, input_size + 1:]

        return dX, dWGRU, dh0


# -------------------
# TEST CASES
# -------------------



def checkSequentialMatchesBatch():
    """ check GRU I/O forward/backward interactions """

    n, b, d = (4, 2, 3)  # sequence length, batch size, hidden size
    input_size = 5
    WGRU = GRU.init(input_size, d)  # input size, hidden size
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)

    print "h0, " , np.shape(h0)

    # sequential forward

    hprev = h0
    caches = [{} for t in xrange(n)]
    Hcat = np.zeros((n, b, d))
    for t in xrange(n):
        xt = X[t:t + 1]
        _, hprev, cache = GRU.forward(xt, WGRU, hprev)
        caches[t] = cache
        Hcat[t] = hprev

    # sanity check: perform batch forward to check that we get the same thing
    H, _, batch_cache = GRU.forward(X, WGRU,h0)
    assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

    # eval loss
    wrand = np.random.randn(*Hcat.shape)
    loss = np.sum(Hcat * wrand)
    dH = wrand

    # get the batched version gradients
    BdX, BdWGRU, Bdh0 = GRU.backward(dH, batch_cache)

    # now perform sequential backward
    dX = np.zeros_like(X)
    dWGRU = np.zeros_like(WGRU)
    dh0 = np.zeros_like(h0)
    dhnext = None
    for t in reversed(xrange(n)):
        dht = dH[t].reshape(1, b, d)
        dx, dWGRUt, dhprev = GRU.backward(dht, caches[t], dhnext)
        dhnext = dhprev


        dWGRU += dWGRUt  # accumulate GRU gradient
        dX[t] = dx[0]
        if t == 0:
            dh0 = dhprev

    # and make sure the gradients match
    print 'Making sure batched version agrees with sequential version: (should all be True)'
    print np.allclose(BdX, dX)
    print np.allclose(BdWGRU, dWGRU)
    print np.allclose(Bdh0, dh0)


def checkBatchGradient():
    """ check that the batch gradient is correct """

    # lets gradient check this beast
    n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
    input_size = 10
    WGRU = GRU.init(input_size, d)  # input size, hidden size
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)


    # batch forward backward
    H, Ht, cache = GRU.forward(X, WGRU, h0)
    wrand = np.random.randn(*H.shape)
    loss = np.sum(H * wrand)  # weighted sum is a nice hash to use I think
    dH = wrand
    dX, dWGRU, dh0 = GRU.backward(dH, cache)

    def fwd():
        h, _, _ = GRU.forward(X, WGRU, h0)
        return np.sum(h * wrand)

    # now gradient check all
    delta = 1e-5
    rel_error_thr_warning = 1e-2
    rel_error_thr_error = 1
    tocheck = [X, WGRU, h0]
    grads_analytic = [dX, dWGRU, dh0]
    names = ['X', 'WGRU', 'h0']
    for j in xrange(len(tocheck)):
        mat = tocheck[j]
        dmat = grads_analytic[j]
        name = names[j]
        # gradcheck
        for i in xrange(mat.size):
            old_val = mat.flat[i]
            mat.flat[i] = old_val + delta
            loss0 = fwd()
            mat.flat[i] = old_val - delta
            loss1 = fwd()
            mat.flat[i] = old_val

            grad_analytic = dmat.flat[i]
            grad_numerical = (loss0 - loss1) / (2 * delta)

            if grad_numerical == 0 and grad_analytic == 0:
                rel_error = 0  # both are zero, OK.
                status = 'OK'
            elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                rel_error = 0  # not enough precision to check this
                status = 'VAL SMALL WARNING'
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                status = 'OK'
                if rel_error > rel_error_thr_warning: status = 'WARNING'
                if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

            # print stats
            print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
                  % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)


if __name__ == "__main__":
    checkSequentialMatchesBatch()
    raw_input('check OK, press key to continue to gradient check')
    checkBatchGradient()
    print 'every line should start with OK. Have a nice day!'