package dl.util;

import static guru.nidi.graphviz.model.Factory.graph;
import static guru.nidi.graphviz.model.Factory.node;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import guru.nidi.graphviz.attribute.Color;
import guru.nidi.graphviz.attribute.Shape;
import guru.nidi.graphviz.attribute.Style;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;
import guru.nidi.graphviz.model.Label;
import guru.nidi.graphviz.model.Link;
import guru.nidi.graphviz.model.Node;

public class TestGraph {

	@Test
	public void test1() throws IOException {
		// Graph g =
		// graph("example1").directed().with(node("a").link(node("b")));
		// Graphviz.fromGraph(g).width(200).render(Format.PNG).toFile(new
		// File("tmp/ex1.png"));

		Node init = node("init"), execute = node("execute"),
				compare = node("compare").with(Shape.RECTANGLE, Style.FILLED, Color.hsv(.7, .3, 1.0)),
				mkString = node("mkString").with(Label.of("make a\nstring")), printf = node("printf");

		Graph g = graph("example2").directed()
				.with(node("main").with(Shape.RECTANGLE).link(
						Link.to(node("parse").link(execute)).with("weight", 8), Link.to(init).with(Style.DOTTED),
						node("cleanup"), Link.to(printf).with(Style.BOLD, Label.of("100 times"), Color.RED)),
						execute.link(graph().with(mkString, printf), Link.to(compare).with(Color.RED)),
						init.link(mkString));

		Graphviz.fromGraph(g).width(900).render(Format.PNG).toFile(new File("tmp/ex2.png"));
	}

}
