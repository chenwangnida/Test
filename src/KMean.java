import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.stat.regression.SimpleRegression;

import com.google.common.collect.Ordering;

public class KMean {

	public static void main(String[] args) {

		// we have a list of our locations we want to cluster. create a
		List<Location> locations = new ArrayList<Location>();
		Random rand = new Random();

		// add some test points
		for (int i = 0; i < 500; i++) {
			double x = rand.nextDouble();
			double y = rand.nextDouble();

			locations.add(new Location(x, y));
		}

		List<LocationWrapper> clusterInput = new ArrayList<LocationWrapper>(locations.size());
		for (Location location : locations)
			clusterInput.add(new LocationWrapper(location));

		// initialize a new clustering algorithm.
		// we use KMeans++ with 10 clusters and 10000 iterations maximum.
		// we did not specify a distance measure; the default (euclidean distance) is
		// used.
		KMeansPlusPlusClusterer<LocationWrapper> clusterer = new KMeansPlusPlusClusterer<LocationWrapper>(4, 10000);
		List<CentroidCluster<LocationWrapper>> clusterResults = clusterer.cluster(clusterInput);

		// output the clusters
		for (int i = 0; i < clusterResults.size(); i++) {
			System.out.println("Cluster " + i);
			SimpleRegression regression = new SimpleRegression();
			double[][] data = new double[clusterResults.get(i).getPoints().size()][2];

			int j = 0;

			for (LocationWrapper locationWrapper : clusterResults.get(i).getPoints()) {
				System.out.println(locationWrapper.getLocation() + ":" + locationWrapper.getLocation().getX()
						+ locationWrapper.getLocation().getY());
				data[j][0] = locationWrapper.getLocation().getX();
				data[j][1] = locationWrapper.getLocation().getY();
				j++;
			}

			regression.addData(data);
			System.out.println();
			System.out.println(regression.getIntercept());
			// displays intercept of regression line

			System.out.println(regression.getSlope());
			// displays slope of regression line

			System.out.println(regression.getSlopeStdErr());
			// displays slope standard error

			for (LocationWrapper locationWrapper : clusterResults.get(i).getPoints()) {

				double y_pre = regression.getSlope() * locationWrapper.getLocation().getX() + regression.getIntercept();
				double diff = Math.pow((y_pre - locationWrapper.getLocation().getY()), 2);
				locationWrapper.setDiff(diff);
			}

			LocationWrapper max = Collections.max(clusterResults.get(i).getPoints(),
					Comparator.comparingDouble(a -> a.diff));
			System.out.println(max.diff);

		}
	}

	public static class LocationWrapper implements Clusterable {
		private double[] points;
		private Location location;
		private double diff;// difference between y and predicted y
		double euclidean_distance;

		public LocationWrapper(Location location) {
			this.location = location;
			this.points = new double[] { location.getX(), location.getY() };
		}

		public Location getLocation() {
			return location;
		}

		public double[] getPoint() {
			return points;
		}

		public double getDiff() {
			return diff;
		}

		public void setDiff(double diff) {
			this.diff = diff;
		}

	}

}
