import numpy as np
import pandas as pd
from typing import List, Tuple
import sklearn


def sort_in_unison(y_true, y_score, cards=None):
  sorted_indices = y_score.argsort()
  y_true_sorted = y_true[sorted_indices][::-1]
  y_score_sorted = y_score[sorted_indices][::-1]
  if cards is None:
    return y_true_sorted, y_score_sorted
  else:
    cards_sorted = cards[sorted_indices][::-1]
    return y_true_sorted, y_score_sorted, cards_sorted


def apply_threshold(y_score, threshold):
  return y_score >= threshold


def count(y_true, type="tx", cards=None):
  if type == "tx":
    return y_true.size
  elif type == "card":
    if cards is not None:
      return np.unique(cards).size
    else:
      raise ValueError(
          "cards values are None while trying to calculate the count for cards")
  else:
    raise ValueError("count for " + str(type) + " is not implemented")


def count_fraud(y_true, type="tx", cards=None):
  if type == "tx":
    return (y_true == 1).sum()
  elif type == "card":
    if cards is not None:
      true_indexes = np.argwhere(y_true == 1)
      fraud_cards = cards[true_indexes].flatten()
      unique_fraud_cards = pd.unique(fraud_cards)
      return unique_fraud_cards.size
    else:
      raise ValueError(
          "cards values are None while trying to calculate the count-fraud for cards")
  else:
    raise ValueError("count-fraud for " + str(type) + " is not implemented")


def true_positive(y_true, y_pred, type="tx", cards=None):
  if type == "tx":
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return tp
  elif type == "card":
    if cards is not None:
      # We are looking for genuine cards detected as fraud
      df_pred = pd.DataFrame.from_dict({
          "pan_ids": cards,
          "y_true": y_true,
          "y_pred": y_pred
      }, orient='columns')
      df_by_pan_ids = df_pred.groupby("pan_ids").max()
      y_true_card = df_by_pan_ids["y_true"].values
      y_pred_card = df_by_pan_ids["y_pred"].values
      fp_cards = (y_true_card == 1) & (y_pred_card == 1)
      return fp_cards.sum()
    else:
      raise ValueError(
          "cards values are None while trying to calculate the true_positive for cards")
  else:
    raise ValueError("true_negative for " + str(type) + " is not implemented")


def true_negative(y_true, y_pred, type="tx", cards=None):
  if type == "tx":
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn
  elif type == "card":
    if cards is not None:
      # We are looking for fraudulent cards not detected (detected as genuine)
      df_pred = pd.DataFrame.from_dict({
          "pan_ids": cards,
          "y_true": y_true,
          "y_pred": y_pred
      }, orient='columns')
      df_by_pan_ids = df_pred.groupby("pan_ids").max()
      y_true_card = df_by_pan_ids["y_true"].values
      y_pred_card = df_by_pan_ids["y_pred"].values
      fn_cards = (y_true_card == 0) & (y_pred_card == 0)
      return fn_cards.sum()
    else:
      raise ValueError(
          "cards values are None while trying to calculate the true_negative for cards")
  else:
    raise ValueError("true_negative for " + str(type) + " is not implemented")


def false_positive(y_true, y_pred, type="tx", cards=None):
  if type == "tx":
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return fp
  elif type == "card":
    if cards is not None:
      # We are looking for genuine cards detected as fraud
      df_pred = pd.DataFrame.from_dict({
          "pan_ids": cards,
          "y_true": y_true,
          "y_pred": y_pred
      }, orient='columns')
      df_by_pan_ids = df_pred.groupby("pan_ids").max()
      y_true_card = df_by_pan_ids["y_true"].values
      y_pred_card = df_by_pan_ids["y_pred"].values
      fp_cards = (y_true_card == 0) & (y_pred_card == 1)
      return fp_cards.sum()
    else:
      raise ValueError(
          "cards values are None while trying to calculate the false_positive for cards")
  else:
    raise ValueError("false_negative for " + str(type) + " is not implemented")


def false_negative(y_true, y_pred, type="tx", cards=None):
  if type == "tx":
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return fn
  elif type == "card":
    if cards is not None:
      # We are looking for fraudulent cards not detected (detected as genuine)
      df_pred = pd.DataFrame.from_dict({
          "pan_ids": cards,
          "y_true": y_true,
          "y_pred": y_pred
      }, orient='columns')
      df_by_pan_ids = df_pred.groupby("pan_ids").max()
      y_true_card = df_by_pan_ids["y_true"].values
      y_pred_card = df_by_pan_ids["y_pred"].values
      fn_cards = (y_true_card == 1) & (y_pred_card == 0)
      return fn_cards.sum()
    else:
      raise ValueError(
          "cards values are None while trying to calculate the false_negative for cards")
  else:
    raise ValueError("false_negative for " + str(type) + " is not implemented")


def amount_saved(y_true, y_pred, amounts, cards=None):
  if cards is not None:
    raise ValueError(
        "The metric 'amount_saved' has not been implemented for cards. It only looks at transactions.")
  return amounts[(y_true == 1) & (y_pred == 1)].sum() - amounts[(y_true == 1) & (y_pred == 0)].sum()


def precision(y_true, y_pred, type="tx", cards=None, reference_ratio=None):
  if type == "tx":
    pi_current = np.count_nonzero(y_true == 1) / len(y_true)
    tp = np.count_nonzero((y_true == 1) & (y_pred == 1))
    fp = np.count_nonzero((y_true == 0) & (y_pred == 1))
  elif type == "card":
    if cards is not None:
      true_indexes = np.argwhere(y_true == 1)
      true_pred_indexes = np.argwhere(y_pred == 1)
      fraud_cards = cards[true_indexes].flatten()
      fraud_pred_cards = cards[true_pred_indexes].flatten()
      unique_cards = pd.unique(cards)
      unique_fraud_cards = pd.unique(fraud_cards)
      unique_fraud_pred_cards = pd.unique(fraud_pred_cards)
      pi_current = len(unique_fraud_cards) / len(unique_cards)
      tp = 0
      fp = 0
      for fraud_pred_card in unique_fraud_pred_cards:
        if fraud_pred_card in unique_fraud_cards:
          tp += 1
        else:
          fp += 1
    else:
      raise ValueError(
          "cards values are None while trying to calculate the precision for cards")
  else:
    raise ValueError("precision for " + str(type) + " is not implemented")

  if (tp + fp) == 0:
    return 0
  elif reference_ratio is not None:
    calibration_factor = pi_current * \
        (1 - reference_ratio) / (reference_ratio * (1 - pi_current))
    return tp / float(calibration_factor * fp + tp)
  else:
    return tp / (fp + tp)


def recall(y_true, y_pred, type="tx", cards=None):
  if type == "tx":
    return sklearn.metrics.recall_score(y_true, y_pred)
  elif type == "card":
    if cards is not None:
      true_indexes = np.argwhere(y_true == 1)
      true_pred_indexes = np.argwhere(y_pred == 1)
      fraud_cards = cards[true_indexes].flatten()
      fraud_pred_cards = cards[true_pred_indexes].flatten()
      unique_fraud_cards = pd.unique(fraud_cards)
      unique_fraud_pred_cards = pd.unique(fraud_pred_cards)
      tp = 0
      for fraud_pred_card in unique_fraud_pred_cards:
        if fraud_pred_card in unique_fraud_cards:
          tp += 1
      if len(unique_fraud_cards) == 0:
        return 0
      return float(tp) / len(unique_fraud_cards)
    else:
      raise ValueError(
          "cards values are None while trying to calculate the recall for cards")
  else:
    raise ValueError("recall for " + str(type) + " is not implemented")


def f_score(y_true, y_pred, type="tx", cards=None, reference_ratio=None):
  p = precision(y_true, y_pred, type=type, cards=cards,
                reference_ratio=reference_ratio)
  r = recall(y_true, y_pred, type=type, cards=cards)
  if p == 0 and r == 0:
    return 0
  return 2 * p * r / (p + r)


def pk(y_true, y_score, rank, type="tx", sorted=False, cards=None):
  assert len(y_score) == len(y_true)
  if type == "tx":
    if sorted:
      y_true_sorted = y_true
    else:
      y_true_sorted, _ = sort_in_unison(y_true, y_score)
    return float(y_true_sorted[:rank].sum()) / rank
  elif type == "card":
    if cards is not None:
      if sorted:
        y_true_sorted = y_true
        cards_sorted = cards
      else:
        y_true_sorted, _, cards_sorted = sort_in_unison(
            y_true, y_score, cards=cards)
      true_indexes = np.argwhere(y_true_sorted == 1)
      fraud_cards = cards_sorted[true_indexes]
      unique_cards = pd.unique(cards_sorted)[:rank]
      cards_fraud_count = 0
      for card in unique_cards:
        if card in fraud_cards:
          cards_fraud_count += 1
      return float(cards_fraud_count) / rank
    else:
      raise ValueError(
          "cards values are None while trying to calculate the pk {} for a card".format(rank))
  else:
    raise ValueError("Pk for " + str(type) + " is not implemented")


def precision_recall_curve(y_true, y_score, type="tx", cards=None):
  if type == "tx":
    return sklearn.metrics.precision_recall_curve(y_true=y_true, probas_pred=y_score)
  elif type == "card":
    if cards is not None:
      df_pred = pd.DataFrame.from_dict({
          "pan_ids": cards,
          "y_true": y_true,
          "y_score": y_score
      }, orient='columns')
      df_pred_by_pan_ids = df_pred.groupby("pan_ids").max()
      y_true_card = df_pred_by_pan_ids["y_true"].values
      y_true_card[y_true_card > 0] = 1
      y_score_card = df_pred_by_pan_ids["y_score"].values
      return sklearn.metrics.precision_recall_curve(y_true=y_true_card, probas_pred=y_score_card)
    else:
      raise ValueError(
          "cards values are None while trying to calculate the PR-AUC for a card")
  else:
    raise ValueError("PR-AUC for {} is not implemented".format(type))


def plot_precision_recall_curve(y_true, y_score, type="tx", cards=None, title=None, plot_save_path=None,
                                precisions_save_path=None, recalls_save_path=None, thresholds_save_path=None):
  precisions, recalls, thresholds = precision_recall_curve(
      y_true, y_score, type=type, cards=cards)

  if precisions_save_path is not None:
    np.save(precisions_save_path, precisions)
  if recalls_save_path is not None:
    np.save(recalls_save_path, recalls)
  if thresholds_save_path is not None:
    np.save(thresholds_save_path, thresholds)

  import matplotlib
  import sys
  if sys.platform != "win32":
    # Force matplotlib to not use any Xwindows backend
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
  plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  if title is None:
    title = "Precision-Recall curve"
  pr_auc_score = pr_auc(y_true, y_score, recall_value=1.0,
                        type=type, cards=cards, sorted=False)
  title = "{} ({:.3f})".format(title, pr_auc_score)
  plt.title(title)
  if plot_save_path is not None:
    plt.savefig(plot_save_path)
    plt.clf()
  else:
    plt.show()


def plot_f_score_curve(y_true, y_score, type="tx", cards=None, title=None, plot_save_path=None,
                       thresholds_save_path=None, precisions_save_path=None, recalls_save_path=None,
                       f_scores_save_path=None):
  precisions, recalls, thresholds = precision_recall_curve(
      y_true, y_score, type=type, cards=cards)
  thresholds = np.append(thresholds, 1.0)
  f_scores = 2 * precisions * recalls / (precisions + recalls)

  if thresholds_save_path is not None:
    np.save(thresholds_save_path, thresholds)
  if precisions_save_path is not None:
    np.save(precisions_save_path, precisions)
  if recalls_save_path is not None:
    np.save(recalls_save_path, recalls)
  if f_scores_save_path is not None:
    np.save(f_scores_save_path, f_scores)

  import matplotlib
  import sys
  if sys.platform != "win32":
    # Force matplotlib to not use any Xwindows backend
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  vecX = np.arange(0, 1.05, 0.1)
  plt.xticks(vecX)
  plt.plot(thresholds, precisions, label="Precision")
  plt.plot(thresholds, recalls, label="Recall")
  plt.plot(thresholds, f_scores, label="F-score")
  plt.xlabel("Threshold")
  plt.ylabel("Metric value")
  plt.ylim([0.0, 1.05])
  if title is None:
    title = "F-score curve"
  plt.title(title)
  plt.legend()
  plt.show()
  if plot_save_path is not None:
    plt.savefig(plot_save_path)
    plt.clf()
  else:
    plt.show()


def average_precision(y_true, y_score, recall_value, sorted=False, reference_ratio=None):
  if sorted:
    y_true_sorted = y_true
    y_score_sorted = y_score
  else:
    y_true_sorted, y_score_sorted = sort_in_unison(y_true, y_score)
  total = 0
  frauds = 0
  p = []
  total_frauds = y_true.sum()

  if reference_ratio is not None:
    pi_current = total_frauds / len(y_true)
    calibration_factor = pi_current * \
        (1 - reference_ratio) / (reference_ratio * (1 - pi_current))
  else:
    calibration_factor = None

  for true, prediction in zip(y_true_sorted, y_score_sorted):
    total += 1
    r = frauds / total_frauds
    if r > recall_value:
      # If the recall becomes greater than 'recall_value', we stop at this point.
      break
    if true == 1:
      frauds += 1
      if calibration_factor is not None:
        fp = float(total - frauds)
        p.append(frauds / float(calibration_factor * fp + frauds))
      else:
        p.append(frauds / float(total))
  return np.sum(p) / total_frauds


def pr_auc(y_true, y_score, recall_value=1.0, type="tx", cards=None, sorted=False, reference_ratio=None):
  if type == "tx":
    return average_precision(y_true, y_score, recall_value, sorted=sorted, reference_ratio=reference_ratio)
  elif type == "card":
    if cards is not None:
      df_pred = pd.DataFrame.from_dict({
          "pan_ids": cards,
          "y_true": y_true,
          "y_score": y_score
      }, orient='columns')
      df_pred_by_pan_ids = df_pred.groupby("pan_ids").max()
      y_true_card = df_pred_by_pan_ids["y_true"].values
      y_true_card[y_true_card > 0] = 1
      y_score_card = df_pred_by_pan_ids["y_score"].values
      return average_precision(y_true_card, y_score_card, recall_value, sorted=sorted, reference_ratio=reference_ratio)
    else:
      raise ValueError(
          "cards values are None while trying to calculate the PR-AUC {} for a card".format(recall_value))
  else:
    raise ValueError("PR-AUC for " + str(type) + " is not implemented")


def pr_auc_from_precision_recall_curve(y_true, y_score, recall_value=1.0, type="tx", cards=None):
  precisions, recalls, thresholds = precision_recall_curve(
      y_true, y_score, type=type, cards=cards)
  if recall_value < 1.0:
    stop_index = np.argmax(recalls >= recall_value)
    return precisions[:stop_index].sum() / len(precisions)
  else:
    return precisions.sum() / len(precisions)
