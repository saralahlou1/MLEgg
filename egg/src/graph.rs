use regex::Regex;
use std::collections::HashMap;
use std::fs::{self, File};

use std::io::Write;
use std::path::Path;

#[derive(Default)]
pub struct Graph {
    nodes: HashMap<i32, Node>,
}

pub struct Node {
    data: String,
    children: Vec<i32>,
}

impl Node {
    pub fn get_data(&self) -> &str {
        &self.data
    }
    pub fn get_children(&self) -> &Vec<i32> {
        &self.children
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
        }
    }
    pub fn get_nodes(&self) -> &HashMap<i32, Node> {
        &self.nodes
    }
    pub fn from_file<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let mut result = Graph::new(); // TODO: rust by example says this is bad
        let contents = fs::read_to_string(path).expect("Should have been able to read the file!");

        let mut lines = contents.lines(); // has to be a separate line because we're borrowing it mutably

        // read the first line to figure out what kind of graph
        let first = lines.next().expect("File seems empty...");
        // find the first word of the first line
        match first.split_whitespace().next().unwrap_or("") {
            // maybe this should be a regex?
            "digraph" => println!("found digraph"),
            _ => panic!("This DOT type isn't supported yet!"),
        }
        // assume this is directed

        // read the rest of the lines until the penultimate one
        // we expect all our node definitions to come first
        let node_regex = Regex::new(r#"^\t(?<id>\d+) \[label="(?<data>.*)"\];$"#).unwrap();
        for node_string in lines.by_ref().take_while(|&line| !line.trim().is_empty()) {
            // this lets us match for a line of pure split_whitespace
            // lines are of the form (ID -[->] [label="DATA"];). parse it into a node
            // i think a regex would be easiest
            println!("testing line");
            let Some(caps) = node_regex.captures(node_string) else {
                println!("node not found in line '{}'", node_string);
                continue;
            }; // maybe this is a better way of stopping the iterator? a take_if would work here

            let id: i32 = caps["id"].parse().unwrap();
            let data = caps["data"].to_owned();
            result.nodes.insert(
                id,
                Node {
                    data,
                    children: Vec::new(),
                },
            );
            println!("node inserted!");
        }
        // then a blank line

        // then all our edges
        let edge_regex = Regex::new(r#"^\t(?<from>\d+) -[->] (?<to>\d+);$"#).unwrap();
        for edge_string in lines {
            // edges look like (ID -[->] ID;)
            let Some(caps) = edge_regex.captures(edge_string) else {
                continue;
            };

            let from: i32 = caps["from"].parse().unwrap();
            let to: i32 = caps["to"].parse().unwrap();
            let from_node = result.nodes.get_mut(&from).unwrap(); // TODO: error checking here -- what if the dot file is malformed?
            from_node.children.push(to);
        }

        // that's it! we're done

        println!("read graph with {} nodes", result.nodes.len());

        return result;
    }
    pub fn to_file<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        // read a file and parse it to dot
        // open the file
        let mut file = match File::create(path) {
            Err(why) => panic!("couldn't create file: {}", why),
            Ok(file) => file,
        };

        file.write_all(b"digraph {\n").expect("couldn't write data");

        // list of nodes
        for (id, node) in &self.nodes {
            file.write_all(format!("\t{} [label=\"{}\"];\n", id, node.data).as_bytes())
                .expect("couldn't write data");
        }
        file.write_all(b"\n").expect("couldn't write data!");
        // for each node, write all children
        for (id, node) in &self.nodes {
            for child in node.get_children() {
                file.write_all(format!("\t{id} -> {child};\n").as_bytes())
                    .expect("couldn't write data");
            }
        }

        file.write_all(b"}").expect("couldn't write data");
        // file auto closed
    }
}
