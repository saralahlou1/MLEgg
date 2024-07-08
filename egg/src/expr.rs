// in mlir-project-test
use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::Graph;
use crate::mlir::MLIR;
use egg::{Id, RecExpr};
use regex::Regex;

pub fn from_dag(graph: &Graph) -> Vec<RecExpr<MLIR>> {
    let mut result_list: Vec<RecExpr<MLIR>> = Vec::new();

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

    println!("Nodes added: {:?}", nodes_added);

    let mut enodes: HashMap<i32, Id> = HashMap::new();

    let mut result: RecExpr<MLIR> = RecExpr::default();

    // Add new language stuff here
    for id in nodes_added.iter().rev() {
        let node = &nodes[id];
        let enode = match node.get_data() {
            "linalg.matmul" => {
                let old_id = result.add(MLIR::Num(*node.get_old_id()));
                result.add(MLIR::MatMul([
                enodes[&node.get_children()[0]],
                enodes[&node.get_children()[1]],
                old_id,
                old_id
            ]))
        },
        "linalg.add" => {
            let old_id = result.add(MLIR::Num(*node.get_old_id()));
            result.add(MLIR::Add([
            enodes[&node.get_children()[0]],
            enodes[&node.get_children()[1]],
            old_id,
            // at first old id = old op id
            old_id
        ]))
        },
            "linalg.dot" => {
                let old_id = result.add(MLIR::Num(*node.get_old_id()));
                result.add(MLIR::Dot([
                enodes[&node.get_children()[0]],
                enodes[&node.get_children()[1]],
                old_id,
                old_id
            ]))
        },
            "linalg.transpose" => {
                let old_id = result.add(MLIR::Num(*node.get_old_id()));
                result.add(MLIR::Transpose([
                enodes[&node.get_children()[0]],
                old_id,
                old_id
            ]))
        },
            "tensor.extract_slice" => {
                let old_id = result.add(MLIR::Num(*node.get_old_id()));
                result.add(MLIR::ExtractSlice([
                enodes[&node.get_children()[0]],
                old_id,
                old_id
            ]))
        },
            // anything else in our language refers to matrices
            _ => {
                // our matrix expr stores the dims of the matrix
                // these will be important for our e-class 
                // analysis, also useful for the cost fn
                let row_id = result.add(MLIR::Num(*node.get_rows()));
                let column_id = result.add(MLIR::Num(*node.get_columns()));
                // not sure if this will be needed to but our matrix expr 
                // will also store the old id from the original dot file
                let old_id = result.add(MLIR::Num(*node.get_old_id()));
                result.add(MLIR::Matrix([row_id, column_id, old_id, old_id]))
            }, 
        };

        enodes.insert(*id, enode);

        // after inserting an expression, we check if the node 
        // corresponded to one of the roots (stored in parents hashset)
        if parents.contains(id) {
            // if so that means that we finished processing one of the 
            // independent expressions. We add the expr to the list
            result_list.push(result.clone());
        }

    }
    // check if the list contains all the expressions
    println!("the extracted expressions are the following: ");
    for expr in &result_list {
        println!("{}", expr);
    }

    result_list
}

pub fn to_dag(expr_list: &Vec<RecExpr<MLIR>>) -> Graph {
    // error check
    for expr in expr_list {
        if !expr.is_dag() {
            panic!("isn't a dag!");
        }
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
    let mut id = 0;

    // for each independant expr we repeat the following
    for expr in expr_list {
        let expr_string = expr.to_string();
        println!("pretty: {}", expr_string);
        let sexp_regex = Regex::new(r"\((?<op>.+?) (?<other>.*)\)").unwrap();
        let mut to_consider: VecDeque<(i32, String)> = VecDeque::new();
        to_consider.push_back((id, expr_string));
        id += 1;

        while !to_consider.is_empty() {
            let next = to_consider.pop_front().unwrap();
            println!("considering {}", next.1);
            // this should always match normally
            let mut children: Vec<i32> = Vec::new();
            if let Some(matches) = sexp_regex.captures(&next.1) {
                println!("found a match!");
                let op = &matches["op"];
                // we initialize the dims to be 0
                // we modify the value if we encounter a matrix
                let mut rows = 0;
                let mut columns = 0;
                let mut data = matches["op"].to_owned();
                let mut old_id = 0;
                let mut old_op_id = 0;

                let other = &matches["other"];

                // check if it's a matrix
                if op == "matrix" && !other.is_empty(){
                    // find the split points -- which are the white spaces
                    let arg_list: Vec<&str> = other.split(" ").collect();
                    // the args for a matrix op are always:
                    // 0 -> rows, 1 -> columns, 2 -> id in old dot graph, 3 -> old id of the op
                    rows = arg_list[0].parse().unwrap();
                    columns = arg_list[1].parse().unwrap();

                    // we store into the data field the id used in the original dot file
                    data = arg_list[2].to_string();

                    old_id = arg_list[2].parse().unwrap();
                    old_op_id = arg_list[3].parse().unwrap();
                }

                // else it should be an operation
                // parse other: get the arguments to parse them
                else if !other.is_empty() {       

                    // we store in a list the arguments of the operation
                    let mut arg_list: Vec<&str> = Vec::new();

                    // find the split points -- which are the white spaces
                    // note that not all white spaces seperate the arguments of the op
                    // ie: when have nested operations
                    let mut paren_count = 0;
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

                    // if we have more, we grab arguments one by one and add them to the list
                    // 'arg' will represent the argument found
                    // 'rest' stores the rest of the args to be processed
                    let mut arg ;
                    let mut rest: &str = other;

                    while break_point != None {
                        // we split at the break point
                        let (_first, _second) = rest.split_at(break_point.unwrap());
                        arg= _first ;
                        // remove the space from the start of _second and update the value of rest
                        rest = &_second[1..];
                        // we push arg to the list
                        arg_list.push(arg);

                        // we process the rest. If there is no more arg, we exit the loop.
                        break_point = None;
                        for (i, c) in rest.chars().enumerate() {
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
                        
                    }
                    // exiting the loop means that there is no more break point
                    // we push the rest to the list which will correspond to an arg
                    arg_list.push(rest);


                    for argument in arg_list {
                        println!(" The arg is: {}", argument);
                        // if it starts with a parenthesis, it's got children
                        // for our current language, this should always be the case
                        // since we handle matrix case alone which by construction
                        // will be the only node to contain terminators
                        if argument.chars().nth(0) == Some('(') {
                            to_consider.push_back((id, argument.to_owned()));
                            children.push(id);
                            id += 1;
                        } 
                        
                        else {
                            // else it's the last arguments which is the old id or old_op_id
                            // they are different only for matrices
                            old_id = argument.parse().unwrap();
                            old_op_id = argument.parse().unwrap();
                            // result.add_node_new(id, argument.to_owned(), vec![], 0, 0, 0);
                        }
                    }

                }
                // we add the node with it's corresponding children and dims
                // the dims will be 0 for anything other than matrices
                // matrices won't have any children
                result.add_node_new(next.0, data, children, rows, columns, old_id, old_op_id);
            } 
            else {
                // change since u added matrices
                println!("We shouldn't get here normally.");
                // result.add_node(next.0, next.1, children)
            }
        }
    }

    result
}
