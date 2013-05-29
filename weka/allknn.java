import java.io.*;
import weka.core.*;
import weka.core.neighboursearch.KDTree;
import weka.core.converters.ConverterUtils.DataSource;

public class allknn {
	public static void main(String args[]) {
		try {
			// Load the Data.
			DataSource source = new DataSource("../data/fisheriris.arff");
			Instances data = source.getDataSet();

			// Choose the last two columns without the class.
			data.deleteAttributeAt(0);
			data.deleteAttributeAt(0);
			data.deleteAttributeAt(2);

			source = new DataSource("../data/knn_query.arff");
			Instances query = source.getDataSet();

			// Calculate the 2-nearest-neighbors.
			KDTree k = new KDTree();
			k.setInstances(data);

			for (int i = 0; i < query.numInstances(); i++) {
				Instances output = k.kNearestNeighbours(query.instance(i), 2);
				// Show the results.
				System.out.println(output);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
