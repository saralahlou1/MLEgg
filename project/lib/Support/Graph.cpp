#include "Support/Graph.h"
#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Value.h>
#include <fstream>
#include <iostream>
#include <string>
#include <regex>

Graph::Node& Graph::add_node(int id, std::string data) {
    nodes.try_emplace(id, Graph::Node{data, std::vector<int>()});
    return nodes.find(id)->second;
}

void Graph::to_file(const std::string& filename) {
    std::string sep = directed ? " -> " : " -- ";

    // is this the best way to do things?
    // TODO: maybe use a pipe
    std::ofstream file;
    file.open(filename);

    // write the file header
    file << (directed ? "digraph" : "graph") << " {\n";

    // write the nodes
    for (auto const& [id, node] : nodes) {
        file << '\t' << id << " [label=\"" << node.data << "\"];\n";
    }

    file << '\n';

    // write the edges
    for (auto const& [id, node] : nodes) {
        for (auto const& other : node.children) {
            file << '\t'<< id << sep << other << ";\n";
        }
    }

    // write the footer
    file << "}\n";

    // clean up
    file.close();
}

Graph Graph::from_file(const std::string& filename) {
    return Graph();
}
