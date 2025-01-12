import logging
import pdb
import time
import torch


def get_cost(w, w_base, prev_pi_base):
    """
    Comptues Cost between w and w_base.
    This method performs cost computation in a way that does not result in
    Runtime Error of CUDA out of memory.
    :param w: 1 x A x 1 x k x k, k could be absent
    :param w_base: N x 1 x B x k x k, k could be absent
    :param prev_pi_base: A x B
    :return: Cost = ((w - w_base).sum() * prev_pi).sum()
    """
    start = time.time()
    is_conv = len(w.size()) > 4
    n = w_base.size(1)
    block_size = n
    while True:
        try:
            idx = 0
            cost_i_arr = []
            while idx < n:
                end = idx + block_size
                if end > n:
                    end = n
                w_i = w_base[:, idx:end]
                cost_arr = []
                for w_item in w:
                    w_item = w_item.unsqueeze(0)
                    diff = (w_item - w_i).pow(2)
                    if is_conv:
                        diff = diff.sum(-1).sum(-1)
                    cost_arr.append((diff * prev_pi_base).sum(-1).sum(-1))
                cost_i_arr.append(torch.cat(cost_arr, dim=0))
                idx += block_size
            end = time.time()
            logging.debug('Time to get cost {} seconds'.format(end - start))
            return torch.cat(cost_i_arr, dim=1)
        except RuntimeError as e:
            logging.debug('Error: with blocksize {}'.format(block_size))
            error = "{}".format(e)
            if error.startswith("CUDA out of memory.") and block_size > 1:
                block_size = block_size // 2
            else:
                print(error)
                raise ImportError(e)
            logging.debug('Trying: with blocksize {}'.format(block_size))


def get_activation_cost(act1, act2):
    """
    Returns the euclidean cost between activations in a way that can be executed within
    the memory constraints.
    :param act1: C1 x D
    :param act2: C2 X D
    :return: C1 X C2
    """
    start_time = time.time()
    total_size = act1.size(0)
    block_size = total_size
    while True:
        logging.debug("Trying blocksize {}".format(block_size))
        try:
            cost_arr = []
            idx = 0
            while idx < total_size:
                end = idx + block_size
                if end > total_size:
                    end = total_size
                act = act1[idx:end]
                idx += block_size
                cost_arr.append((act.unsqueeze(1) - act2.unsqueeze(0)).pow(2).sum(-1))
            end_time = time.time()
            logging.debug("Execution time {} s".format(end_time - start_time))
            return torch.cat(cost_arr, dim=0)
        except RuntimeError as e:
            logging.debug('Error: with blocksize {}'.format(block_size))
            error = "{}".format(e)
            if error.startswith("CUDA out of memory.") and block_size > 1:
                block_size = block_size // 2
            else:
                print(error)
                raise ImportError(e)


################## TESTS ###################

def test_get_cost_helper(c, n, m, k=None):
    if k is None:
        w = torch.rand(n, 1, c, 1)
        w_base = torch.rand(1, m, 1, c)
    else:
        w = torch.rand(n, 1, c, 1, k, k)
        w_base = torch.rand(1, m, 1, c, k, k)
    prev_pi_base = torch.rand(c, c)
    if torch.cuda.is_available():
        w = w.cuda()
        w_base = w_base.cuda()
        prev_pi_base = prev_pi_base.cuda()
    start = time.time()
    cost = get_cost(w, w_base, prev_pi_base)
    end = time.time()
    print('Runtime {} seconds'.format(end - start))
    assert cost.size(0) == n
    assert cost.size(1) == m


def test_get_cost():
    test_get_cost_helper(c=128, n=128, m=128)
    test_get_cost_helper(c=256, n=256, m=256)
    test_get_cost_helper(c=512, n=512, m=512)
    test_get_cost_helper(c=1024, n=1024, m=1024)

    test_get_cost_helper(c=128, n=128, m=128, k=3)
    test_get_cost_helper(c=256, n=256, m=256, k=3)
    test_get_cost_helper(c=512, n=512, m=512, k=3)
    test_get_cost_helper(c=512, n=512, m=512, k=7)


def test_get_activation_cost_helper(n, m, d):
    act1 = torch.rand(n, d)
    act2 = torch.rand(m, d)
    if torch.cuda.is_available():
        act1 = act1.cuda()
        act2 = act2.cuda()
    cost = get_activation_cost(act1, act2)
    assert cost.size(0) == n
    assert cost.size(1) == m


def test_get_activation_cost():
    test_get_activation_cost_helper(n=128, m=128, d=256*16*16)
    test_get_activation_cost_helper(n=512, m=512, d=256*32*32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_get_cost()
    test_get_activation_cost()
