use regex::Regex;
use std::fs;
use std::path;

pub struct Graph {}

impl Graph {
    pub fn from_file(path: &path::Path) -> petgraph::graph::DiGraph<(i32, &str), ()> {
        let mut result = petgraph::graph::DiGraph::<(i32, &str), ()>::new();
        let contents = fs::read_to_string(path).expect("Should have been able to read the file!");

        let mut lines = contents.lines(); // has to be a separate line because we're borrowing it mutably

        // read the first line to figure out what kind of graph
        let first = lines.next().expect("File seems empty...");
        // find the first word of the first line
        match first.split_whitespace().next().unwrap_or("") {
            // maybe this should be a regex?
            "digraph" => (),
            _ => panic!("This DOT type isn't supported yet!"),
        }
        // assume this is directed

        // read the rest of the lines until the penultimate one
        // we expect all our node definitions to come first
        let node_regex = Regex::new(r#"^(?<id>\d+) -[->] \[label="(?<data>.*)"\];$"#).unwrap();
        for node_string in lines.by_ref().take_while(|&line| !line.trim().is_empty()) {
            // this lets us match for a line of pure split_whitespace
            // lines are of the form (ID -[->] [label="DATA"];). parse it into a node
            // i think a regex would be easiest
            let Some(caps) = node_regex.captures(node_string) else {
                continue;
            }; // maybe this is a better way of stopping the iterator? a take_if would work here

            let id = caps["id"].parse().unwrap();
            let data = &caps["data"];
            result.add_node((id, data));
        }
        // then a blank line

        // then all our edges
        let edge_regex = Regex::new(r#"^(?<from>\d+) -[->] (?<to>\d+);$"#).unwrap();
        for edge_string in lines {
            // edges look like (ID -[->] ID;)
            let Some(caps) = edge_regex.captures(edge_string) else {
                continue;
            };

            // TODO: ask asa what ownership is about. i don't get why this needs an ampersand but the other one doesn't
            let from = &caps["from"].parse().unwrap();
            let to = &caps["to"].parse().unwrap();
            let from_node = result.nodes.get_mut(from).unwrap(); // TODO: error checking here -- what if the dot file is malformed?
            let to_node = result.nodes.get(to).unwrap();
            from_node.children.push(to_node.id);
        }

        // that's it! we're done

        return result;
    }
    pub fn to_file() {}
}
