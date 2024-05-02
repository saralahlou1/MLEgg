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
    std::ifstream file(filename);

    // error check
    if (!file.is_open()) {
        // error!
        return Graph();
    }
    
    Graph output = Graph();

    // same reading algo:
    // read first line for directedness
    std::string line;

    if (!std::getline(file, line)) {
        // error!
    }
    // assume directed
    output.directed = true;
    // while there isn't a newline: add nodes
    std::regex node_regex("^\t(\\d+) \\[label=\"(.*)\"\\];$");
    std::smatch base_match;
    while (std::getline(file, line) && !line.empty()) {
        // lines look like ([whitespace]<id> [label="<data>"];)
        // c++ doesn't support capture groups :(
        if (std::regex_match(line, base_match, node_regex)) {
            if (base_match.size() > 2) {
                int id = std::stoi(base_match[1].str());
                std::string data = base_match[2].str();
                std::cout << "id: " << id << ", data: " << data << "\n";
                output.add_node(id, data);
            }
        }
    }

    // after the newline: add children
    std::regex edge_regex("^\t(\\d+) -[->] (\\d+);$");
    while (std::getline(file, line) && line != "}") {
        // lines look like ([whitespace]<parent> -[->] <child>;)
        if (std::regex_match(line, base_match, edge_regex)) {
            if (base_match.size() > 2) {
                int from = std::stoi(base_match[1].str());
                int to = std::stoi(base_match[2].str());
                std::cout << "from: " << from << ", to: " << to << "\n";
                auto& from_node = output.nodes.find(from)->second; // not error checked!
                from_node.children.push_back(to);
            }
        }
    }

    file.close();
    
    return output;
}
