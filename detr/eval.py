
def eval_pred(cls_pred, cids_gt, query_pos_mask):
    tp_mask = (cls_pred == cids_gt) * query_pos_mask
    query_neg_mask = 1 - query_pos_mask
    tp = tp_mask.sum()
    n_pos = query_pos_mask.sum()
    n_neg = query_neg_mask.sum()
    n_cls = n_pos + n_neg
    tn = ((cls_pred == cids_gt) * query_neg_mask).sum()
    fn = n_pos - tp
    prec = tp / (tp + n_neg - tn)
    accu = (tp + tn) / (n_cls + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * prec * recall / (prec + recall)
    return accu, recall, f1, tp


def eval_pred2(cls_pred, cids_gt):
    eq = (cls_pred == cids_gt) * 1
    tp_mask = eq * (cids_gt > 0)
    tp = tp_mask.sum()
    recall = tp / ((cids_gt > 0).sum() + 1e-5)
    accu = eq.sum() / cls_pred.shape[0]
    return accu, recall
