import cv2
import csv
import numpy as np
import random
from imgman_dsl import *
from typing import Any, List, Dict, Set, Tuple

map_outputs = {}


def is_contained(bbox1, bbox2):
    left1, right1, top1, bottom1 = bbox1
    left2, right2, top2, bottom2 = bbox2
    return left1 > left2 and top1 > top2 and bottom1 < bottom2 and right1 < right2

def is_contained2(bbox1, bbox2):
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2
    return left1 > left2 and top1 > top2 and bottom1 < bottom2 and right1 < right2

def eval_map(map_extr: Map, details: Dict[str, Dict[str, Any]], rec: bool = True, output_dict={}, eval_cache={}) -> Set[str]:
    if rec:
        objs = eval_extractor(map_extr.extractor, details, output_dict=output_dict, eval_cache=eval_cache)
        rest = eval_extractor(map_extr.restriction, details, output_dict=output_dict, eval_cache=eval_cache)
    else:
        objs = map_extr.extractor
        rest = map_extr.restriction
    mapped_objs = set()
    if isinstance(map_extr.position, GetPrev):
        # The idea: for each target obj we extract, we need to identify
        # the obj who right boundary is as close to the target right boundary
        # as possible, without being greater.
        for target_obj_id in objs:
            pos = details[target_obj_id]["ObjPosInImgLeftToRight"]
            cur_obj_id = None
            cur_pos = None
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                # if obj_id == target_obj_id:
                #     continue
                if details_map["ObjPosInImgLeftToRight"] >= pos:
                    continue
                if obj_id not in rest:
                    continue
                if cur_pos is None or details_map["ObjPosInImgLeftToRight"] > cur_pos:
                    cur_obj_id = obj_id
                    cur_pos = details_map["ObjPosInImgLeftToRight"]
            if cur_obj_id is not None:
                mapped_objs.add(cur_obj_id)
    elif isinstance(map_extr.position, GetNext):
        for target_obj_id in objs:
            pos = details[target_obj_id]["ObjPosInImgLeftToRight"]
            cur_obj_id = None
            cur_pos = None
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                # if obj_id == target_obj_id:
                #     continue
                if details_map["ObjPosInImgLeftToRight"] <= pos:
                    continue
                if obj_id not in rest:
                    continue
                if cur_pos is None or details_map["ObjPosInImgLeftToRight"] < cur_pos:
                    cur_obj_id = obj_id
                    cur_pos = details_map["ObjPosInImgLeftToRight"]
            if cur_obj_id is not None:
                mapped_objs.add(cur_obj_id)
    elif isinstance(map_extr.position, GetBelow):
        for target_obj_id in objs:
            target_left, target_top, target_right, _ = details[target_obj_id]["Loc"]
            cur_obj_id = None
            cur_top = None
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                if obj_id == target_obj_id:
                    continue
                if obj_id not in rest:
                    continue
                left, top, right, bottom = details_map["Loc"]
                if top < target_top:
                    continue
                if right < target_left or left > target_right:
                    continue
                if cur_top is None or top < cur_top:
                    cur_top = top
                    cur_obj_id = obj_id
            if cur_obj_id is not None:
                mapped_objs.add(cur_obj_id)
    elif isinstance(map_extr.position, GetAbove):
        for target_obj_id in objs:
            target_left, target_top, target_right, target_bottom = details[target_obj_id]["Loc"]
            cur_obj_id = None
            cur_bottom = None
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                if obj_id == target_obj_id:
                    continue
                if obj_id not in rest:
                    continue
                left, top, right, bottom = details_map["Loc"]
                if bottom > target_bottom:
                    continue
                if right < target_left or left > target_right:
                    continue
                if cur_bottom is None or bottom > cur_bottom:
                    cur_bottom = bottom
                    cur_obj_id = obj_id
            if cur_obj_id is not None:
                mapped_objs.add(cur_obj_id)
    elif isinstance(map_extr.position, GetLeft):
        for target_obj_id in objs:
            target_left, target_top, target_right, target_bottom = details[target_obj_id]["Loc"]
            cur_obj_id = None
            cur_left = None
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                if obj_id == target_obj_id:
                    continue
                if obj_id not in rest:
                    continue
                left, top, right, bottom = details_map["Loc"]
                if left > target_left:
                    continue
                if bottom < target_top or top > target_bottom:
                    continue
                if cur_left is None or left > cur_left:
                    cur_left = left
                    cur_obj_id = obj_id
            if cur_obj_id is not None:
                mapped_objs.add(cur_obj_id)
    elif isinstance(map_extr.position, GetRight):
        for target_obj_id in objs:
            target_left, target_top, target_right, target_bottom = details[target_obj_id]["Loc"]
            cur_obj_id = None
            cur_left = None
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                if obj_id == target_obj_id:
                    continue
                if obj_id not in rest:
                    continue
                left, top, right, bottom = details_map["Loc"]
                if left < target_left:
                    continue
                if bottom < target_top or top > target_bottom:
                    continue
                if cur_left is None or left < cur_left:
                    cur_left = left
                    cur_obj_id = obj_id
            if cur_obj_id is not None:
                mapped_objs.add(cur_obj_id)
    elif isinstance(map_extr.position, GetContains):
        for target_obj_id in objs:
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                if obj_id == target_obj_id:
                    continue
                if obj_id not in rest:
                    continue
                if is_contained2(details_map["Loc"], details[target_obj_id]["Loc"]):
                    mapped_objs.add(obj_id)
    elif isinstance(map_extr.position, GetIsContained):
        for target_obj_id in objs:
            for obj_id, details_map in details.items():
                if details_map["ImgIndex"] != details[target_obj_id]["ImgIndex"]:
                    continue
                if obj_id == target_obj_id:
                    continue
                if obj_id not in rest:
                    continue
                if is_contained2(details[target_obj_id]["Loc"], details_map["Loc"]):
                    mapped_objs.add(obj_id)
    if rec:
        return mapped_objs
    else:
        return mapped_objs


def eval_extractor(
    extractor: Extractor, details: Dict[str, Dict[str, Any]], rec: bool = True, output_dict={}, eval_cache=None
):  # -> Set[dict[str, str]]:
    if output_dict and extractor.val is not None:
        return output_dict[extractor.val]
    if eval_cache and str(extractor) in eval_cache:
        return eval_cache[str(extractor)]
    if isinstance(extractor, Map):
        res = eval_map(extractor, details, rec, output_dict, eval_cache)
    elif isinstance(extractor, IsFace):
        # list of all face ids in target image
        res = {obj for obj in details.keys() if details[obj]["Type"] == "Face"}
    elif isinstance(extractor, IsText):
        res = {obj for obj in details.keys() if details[obj]["Type"] == "Text"}
    elif isinstance(extractor, GetFace):
        objs = set()
        for (obj_id, obj_details) in details.items():
            if obj_details["Type"] != "Face":
                continue
            if obj_details["Index"] == extractor.index:
                objs.add(obj_id)
        res = objs
    elif isinstance(extractor, IsObject):
        objs = set()
        for (obj_id, obj_details) in details.items():
            if obj_details["Type"] != "Object":
                continue
            if obj_details["Name"] == extractor.obj:
                objs.add(obj_id)
        res = objs
    elif isinstance(extractor, MatchesWord):
        objs = set()
        for (obj_id, obj_details) in details.items():
            if obj_details["Type"] != "Text":
                continue
            if obj_details["Text"].lower() == extractor.word.lower():
                objs.add(obj_id)
        res = objs
    elif isinstance(extractor, BelowAge):
        objs = set()
        for (obj_id, obj_details) in details.items():
            if obj_details["Type"] != "Face":
                continue
            if extractor.age > obj_details["AgeRange"]["Low"]:
                objs.add(obj_id)
        res = objs
    elif isinstance(extractor, AboveAge):
        objs = set()
        for (obj_id, obj_details) in details.items():
            if obj_details["Type"] != "Face":
                continue
            if extractor.age < obj_details["AgeRange"]["High"]:
                objs.add(obj_id)
        res = objs
    elif isinstance(extractor, Union):
        if rec:
            res = set()
            for sub_extr in extractor.extractors:
                res = res.union(eval_extractor(sub_extr, details, rec, output_dict, eval_cache))
            res = res
        else:
            res = set()
            for sub_extr in extractor.extractors:
                res = res.union(sub_extr.objs)
            res = res
    elif isinstance(extractor, Intersection):
        if rec:
            res = set(details.keys())
            for sub_extr in extractor.extractors:
                res = res.intersection(eval_extractor(sub_extr, details, rec, output_dict, eval_cache))
        else:
            res = set()
            for sub_extr in extractor.extractors:
                res = res.intersection(sub_extr.objs)
    elif isinstance(extractor, Complement):
        # All objs in target image except those extracted
        if rec:
            extracted_objs = eval_extractor(extractor.extractor, details, rec, output_dict, eval_cache)
            res = details.keys() - extracted_objs
        else:
            res = details.keys() - set(extractor.extractor.objs)
    elif (
        isinstance(extractor, IsPhoneNumber)
        or isinstance(extractor, IsPrice)
        or isinstance(extractor, IsSmiling)
        or isinstance(extractor, EyesOpen)
        or isinstance(extractor, MouthOpen)
    ):
        res = {obj for obj in details if str(extractor) in details[obj]}
    else:
        print(extractor)
        raise Exception
    if eval_cache:
        eval_cache[str(extractor)] = res
    return res