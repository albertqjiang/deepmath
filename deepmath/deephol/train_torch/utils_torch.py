from string import whitespace
import torch


def vocab_table_from_file(filename, reverse=False):
    with open(filename, 'r') as f_hand:
        keys = [s.strip() for s in f_hand.readlines()]
        values = range(len(keys))
        return {
            key: value for key, value in zip(keys, values)
        }


"""
    S-expression to AST parser created by Paul Bonser
"""
atom_end = set('()"') | set(whitespace)


def parse(sexp):
    stack, i, length = [[]], 0, len(sexp)
    while i < length:
        c = sexp[i]

        # print(c, stack)
        reading = type(stack[-1])
        if reading == list:
            if   c == '(': stack.append([])
            elif c == ')':
                stack[-2].append(stack.pop())
                # if stack[-1][0] == ("'",): stack[-2].append(stack.pop())
            elif c == '"': stack.append('')
            # elif c == "'": stack.append([("'",)])
            elif c in whitespace: pass
            else: stack.append((c,))
        elif reading == str:
            if   c == '"':
                stack[-2].append(stack.pop())
                # if stack[-1][0] == ("'",): stack[-2].append(stack.pop())
            elif c == '\\':
                i += 1
                stack[-1] += sexp[i]
            else: stack[-1] += c
        elif reading == tuple:
            if c in atom_end:
                atom = stack.pop()
                if atom[0][0].isdigit(): stack[-1].append(eval(atom[0]))
                else: stack[-1].append(atom)
                # if stack[-1][0] == ("'",): stack[-2].append(stack.pop())
                continue
            else: stack[-1] = ((stack[-1][0] + c),)
        i += 1
    return stack.pop()


def ast_to_connections(ast, nodes=None):
    if len(ast) == 0:
        return None, [], []
    if nodes is None:
        nodes = list()
    this_node = ast[0]
    # assert isinstance(this_node, tuple)
    # assert len(this_node) == 1
    node_index = len(nodes)
    if isinstance(this_node, tuple):
        nodes.append(this_node[0])
    elif isinstance(this_node, str):
        nodes.append(this_node)
    elif isinstance(this_node, int):
        nodes.append(str(this_node))
    else:
        print(type(this_node))
        print(this_node)
        raise NotImplementedError
    connections = list()
    for that_node in ast[1:]:
        that_node_index, that_node_connections, _ = ast_to_connections(that_node, nodes)
        connections.append([node_index, that_node_index])
        connections.extend(that_node_connections)

    return node_index, connections, nodes


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
