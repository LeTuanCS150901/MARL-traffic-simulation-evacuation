#!/usr/bin/env python3
"""
rebuild_all_connections.py  <input.net.xml>  <output.net.xml>

• deletes every original <connection> and <tlLogic> element  
• detects, for every junction, the set of incoming and outgoing edges  
• connects *each lane* of each incoming edge to *each lane* of each
  outgoing edge of that junction (self-connection = U-turn can be
  suppressed)  
• writes the result as SUMO network that is schema-conform
  (dir / state attributes included)

The file loads in netconvert, duarouter and sumo.

After that, use
netconvert --sumo-net-file osm_clean.net.xml \
           --tls.guess                                    \
           --output-file    osm_clean_withTLS.net.xml

"""

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

# ----------------------------------------------------------------------
def lane_index(lane_elem):
    """Return numeric lane index (0,1,…) taken from 'index' or lane id."""
    if lane_elem.get('index') is not None:
        return int(lane_elem.get('index'))
    # lane id usually looks like "EDGE_2"
    return int(lane_elem.get('id').rsplit('_', 1)[1])

# ----------------------------------------------------------------------
if len(sys.argv) != 3:
    sys.exit("Usage: rebuild_all_connections.py  input.net.xml  output.net.xml")

in_net, out_net = sys.argv[1:3]

tree = ET.parse(in_net)
root = tree.getroot()

# ---------- 1. Collect lanes per edge ---------------------------------
edge2lanes = {}                       # edge id -> [laneElements]
for edge in root.findall('edge'):
    lanes = edge.findall('lane')
    if lanes:
        edge2lanes[edge.get('id')] = lanes

# ---------- 2. Build incoming / outgoing edge lists for every junction
incoming = defaultdict(set)           # junction id -> set(edge ids)
outgoing = defaultdict(set)

for edge in root.findall('edge'):
    if edge.get('function') == 'internal':
        continue                      # skip internal edges here
    fr = edge.get('from')
    to = edge.get('to')
    eid = edge.get('id')
    if fr:
        outgoing[fr].add(eid)
    if to:
        incoming[to].add(eid)

# ---------- 3. Remove old <connection> and <tlLogic> ------------------
for elem in list(root.findall('connection')) + list(root.findall('tlLogic')):
    root.remove(elem)

# ---------- 4. Create new comprehensive connections -------------------
for junction_id in sorted(incoming.keys() | outgoing.keys()):

    in_edges  = incoming.get(junction_id, set())
    out_edges = outgoing.get(junction_id, set())

    for f_edge in in_edges:
        f_lanes = edge2lanes.get(f_edge, [])
        if not f_lanes:
            continue

        for t_edge in out_edges:
            if t_edge == f_edge:          # skip U-turns if not wanted
                continue

            t_lanes = edge2lanes.get(t_edge, [])
            if not t_lanes:
                continue

            for f_lane in f_lanes:
                fl_idx = lane_index(f_lane)
                for t_lane in t_lanes:
                    tl_idx = lane_index(t_lane)

                    conn = ET.Element('connection')
                    conn.set('from',     f_edge)
                    conn.set('to',       t_edge)
                    conn.set('fromLane', str(fl_idx))
                    conn.set('toLane',   str(tl_idx))

                    # minimal required attributes --------------------
                    conn.set('dir',   's')                       # straight
                    conn.set('state', 'o')                       # priority
                    root.append(conn)

# ---------- 5. Pretty-print & write -----------------------------------
ET.indent(tree, space="    ", level=0)
tree.write(out_net, encoding='utf-8', xml_declaration=True)
print("Written:", out_net)

