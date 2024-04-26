use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::Graph;
use crate::mlir::MLIR;
use egg::{Id, RecExpr};

pub fn from_dag(graph: &Graph) -> RecExpr<MLIR> {
    let mut result: RecExpr<MLIR> = RecExpr::default();

    let nodes = graph.get_nodes();
    println!("{} nodes found", nodes.len());

    // find roots of input graph
    // a root is a node which is the child of no other
    let mut children: HashSet<i32> = HashSet::new();
    for (_, node) in nodes {
        for child in node.get_children() {
            children.insert(child.to_owned()); // TODO: is this right? ask a rustacean
        }
    }
    println!("roots found");
    // difference between all nodes and children is parents
    let all_nodes: HashSet<i32> = nodes.keys().cloned().collect();
    let parents = &all_nodes - &children;

    println!("parents: {:?}", parents);

    // iterate through parents, and BFS
    let mut nodes_seen: VecDeque<i32> = VecDeque::new();
    let mut nodes_added: VecDeque<i32> = VecDeque::new();
    nodes_seen.extend(parents.iter());

    while !nodes_seen.is_empty() {
        let current_id = &nodes_seen.pop_front().unwrap(); // this will never be None
        let current = &nodes[current_id];
        // lambda?
        for child in current.get_children() {
            nodes_seen.push_back(child.to_owned());
        }
        nodes_added.push_back(current_id.to_owned());
    }

    let mut enodes: HashMap<i32, Id> = HashMap::new();

    // Add new language stuff here
    for id in nodes_added.iter().rev() {
        let node = &nodes[id];
        let enode = match node.get_data() {
            "linalg.matmul" => result.add(MLIR::MatMul([
                enodes[&node.get_children()[0]],
                enodes[&node.get_children()[1]],
            ])),
            _ => result.add(MLIR::Var(*id)), // to_owned()?
        };

        enodes.insert(*id, enode);
        println!("inserted");
    }

    return result;
}

pub fn to_dag(expr: &RecExpr<MLIR>) -> Graph {
    // error check
    if !expr.is_dag() {
        panic!("isn't a dag!");
    }

    let mut result: Graph = Graph::new();

    // construct the graph
    // i don't know what to do so.

    return result;
}
