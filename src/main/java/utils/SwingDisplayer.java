package utils;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import javax.swing.JFrame;

public class SwingDisplayer {

	public static void displayImages(List<BufferedImage> images, int win, int unit) {
		Consumer<List<BufferedImage>> displayer = x -> {
			try {
				class MyCanvas extends Canvas {
					private static final long serialVersionUID = 1L;

					public void paint(Graphics g) {
						int x = 0;
						int y = -1;
						BufferedImage img;
						for (int k = 0; k < images.size(); k++) {
							if (x % win == 0) {
								y++;
							}
							img = images.get(k);
							g.drawImage(img, (x % win) * (unit + 5), (y % win) * (unit + 5), this);
							x++;
						}
					}
				}
				MyCanvas m = new MyCanvas();
				JFrame f = new JFrame();
				f.add(m);
				f.setSize((win + 1) * (unit + 5), (win + 1) * (unit + 5));
				f.setVisible(true);
				Thread.sleep(10000);
				f.dispose();
			} catch (Exception e) {
				e.printStackTrace();
			}
		};
		displayer.accept(images);
	}

	public static void displayImages2(List<int[][]> images, int win, int unit) {
		List<BufferedImage> lst = new ArrayList<>();
		for (int[][] im : images) {
			BufferedImage image0 = new BufferedImage(im.length, im[0].length, BufferedImage.TYPE_USHORT_GRAY);
			for (int i = 0; i < im.length; i++) {
				for (int j = 0; j < im[i].length; j++) {
					int a = im[i][j];
					Color newColor = new Color(a, a, a);
					image0.setRGB(j, i, newColor.getRGB());
				}
			}
			lst.add(image0);
		}
		displayImages(lst, win, unit);
	}
}
