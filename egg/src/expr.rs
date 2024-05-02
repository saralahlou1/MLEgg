use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::Graph;
use crate::mlir::MLIR;
use egg::{Id, RecExpr};
use regex::Regex;

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

    result
}

pub fn to_dag(expr: &RecExpr<MLIR>) -> Graph {
    // error check
    if !expr.is_dag() {
        panic!("isn't a dag!");
    }

    let mut result = Graph::new();

    // construct the graph
    // i don't know what to do so.
    // parse the s-expr.
    // let's solve this with regular expressions!
    // queue with s-expressions?
    // pop head. match parens, name of node, rest (greedy).
    // go through rest, assuming that each is 2 arguments
    // i think we can cheat based on space after matching parentheses
    let expr_string = expr.to_string();
    println!("pretty: {}", expr_string);
    let sexp_regex = Regex::new(r"\((?<op>.+?) (?<other>.*)\)").unwrap();
    let mut to_consider: VecDeque<(i32, String)> = VecDeque::new();
    let mut id = 0;
    to_consider.push_back((id, expr_string));
    id += 1;

    while !to_consider.is_empty() {
        let next = to_consider.pop_front().unwrap();
        println!("considering {}", next.1);
        // i mean, this should always work
        let mut children: Vec<i32> = Vec::new();
        if let Some(matches) = sexp_regex.captures(&next.1) {
            println!("found a match!");
            // parse other
            // split it (assume 2 args)
            let other = &matches["other"];
            // first check if it's empty
            if !other.is_empty() {
                // find the split point -- which is the first space
                let mut paren_count = 0;
                // this feels gross and like there should be a better way
                let mut break_point: Option<usize> = None;
                for (i, c) in other.chars().enumerate() {
                    match c {
                        '(' => paren_count += 1,
                        ')' => paren_count -= 1,
                        ' ' if paren_count == 0 => {
                            break_point = Some(i);
                            break;
                        }
                        _ => {}
                    }
                }
                let (_first, _second) = other.split_at(break_point.unwrap()); // i know this is bad but i'm assuming 2 args. i just want this to work
                let (first, second) = (_first, &_second[1..]);

                // if it starts with a parenthesis, it's got children
                if first.chars().nth(0) == Some('(') {
                    to_consider.push_back((id, first.to_owned()));
                } else {
                    result.add_node(id, first.to_owned(), vec![]);
                }
                children.push(id);
                id += 1;

                if second.chars().nth(0) == Some('(') {
                    to_consider.push_back((id, second.to_owned()));
                } else {
                    result.add_node(id, second.to_owned(), vec![]);
                }
                children.push(id);
                id += 1;
                println!("first: {first}, second: {second}");
            }
            // note that we need to make this node
            result.add_node(next.0, matches["op"].to_owned(), children);
        } else {
            println!("why isn't there a match?");
        }
    }

    result
}
