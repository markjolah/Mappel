# mappel/methods.py
#
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
#
# Map the C++ StatsT str->double datatype to a more structured representation in Python
# process_debug_stats also extracts the backtrack_idxs.
#
#
import numpy as np
import re
import struct


_num_regex = re.compile(r"^num|\.num|Num[A-Z]+|Size|Dimensions")
_seed_regex = re.compile(r"\.seed$")
_backtrack_idx_regexp = re.compile(r"^(?P<name>backtrack_idxs)\.(?P<idx>\d+)$")
_vector_regexp = re.compile(r"^(?P<name>\w+)\.(?P<idx>\d+)$")
_dict_regexp = re.compile(r"^(?P<name>\w+)\.(?P<key>\w+)$")


def _double_to_int64(d):
    """
    Turn a double into an int64_t bit for bit.  This allows unpacking an int64_t stored in a double
    to allow a homogeneous data structure of only doubles to be returned from C++
    """
    return struct.unpack("Q",struct.pack("d",d))[0]

def _process_stats_make_integers(stats):
    """
    Convert integers back to ints.
    Deals with seed values bit-for-bit to preserve true uint64_t via the double representation.
    """
    new_stats = stats
    #convert all interger type values to ints using the name.  These are below 2^52 so double to int conversion is exact.
    new_stats.update((k,int(v)) for (k,v) in stats.items() if re.search(_num_regex,k))
    #convert all .seed values to int64 bit-for-bit.  This preserves integer seeds above 2^52
    new_stats.update((k,_double_to_int64(v)) for (k,v) in stats.items() if re.search(_seed_regex,k))
    #conver backtrack_idxs values
    new_stats.update((k,bool(v)) for (k,v) in stats.items() if re.match(_backtrack_idx_regexp,k))
    return new_stats

def process_stats(stats):
    """
    Unpack a string->double dict into a richer structure of vectors and sub-dicts.
    Convert integers back to ints.
    Deals with seed values bit-for-bit to preserve true uint64_t via the double representation.
    """
    #Correct types
    stats = _process_stats_make_integers(stats);
    #Collapse vector and dictionary items
    new_stats={};
    vector_idxs={}
    dict_keys={}
    for (k,v) in stats.items():
        Mvec = re.match(_vector_regexp, k)
        if Mvec:
            name = Mvec.group("name")
            idx = int(Mvec.group("idx"))
            if name not in vector_idxs:
                vector_idxs[name]=[idx]
            else:
                vector_idxs[name].append(idx)
            continue
        Mdict = re.match(_dict_regexp, k)
        if Mdict:
            name = Mdict.group("name")
            key = Mdict.group("key")
            if name not in new_stats:
                new_stats[name]={key:v}
            else:
                new_stats[name][key]=v;
            continue
        new_stats[k]=v #Add scalar value
        
    for (name,idxs) in vector_idxs.items():
        new_stats[name] = np.array([ stats["%s.%i"%(name,idx)] for idx in sorted(idxs)])
    return new_stats

def process_estimator_debug_stats(stats):
    """
    Deal with the backtrack_idxs properly
    """
    new_stats = process_stats(stats)
    if "debugIterative" in new_stats:
        del new_stats["debugIterative"]
    return new_stats
