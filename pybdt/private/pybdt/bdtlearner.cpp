// bdtlearner.cpp

#include <Python.h>

#include "bdtlearner.hpp"
#include "np.hpp"

#include "notifier.hpp"

#include <iostream>

#include <limits>
#include <boost/make_shared.hpp>
#include <boost/pointer_cast.hpp>


using namespace std;
using namespace boost;

BDTLearner::BDTLearner (const vector<string>& feature_names,
                        const string& weight_name)
: Learner (feature_names, weight_name)
{
    set_defaults ();
}

BDTLearner::BDTLearner (const vector<string>& feature_names,
                        const string& sig_weight_name,
                        const string& bg_weight_name)
: Learner (feature_names, sig_weight_name, bg_weight_name)
{
    set_defaults ();
}


BDTLearner::BDTLearner (const boost::python::list& feature_names,
                        const string& weight_name)
: Learner (feature_names, weight_name)
{
    set_defaults ();
}

BDTLearner::BDTLearner (const boost::python::list& feature_names,
                        const string& sig_weight_name,
                        const string& bg_weight_name)
: Learner (feature_names, sig_weight_name, bg_weight_name)
{
    set_defaults ();
}


BDTLearner::~BDTLearner ()
{
}

boost::shared_ptr<DTLearner>
BDTLearner::dtlearner ()
{
    return m_dtlearner;
}

double
BDTLearner::beta () const {
    return m_beta;
}

void
BDTLearner::beta (double beta) {
    m_beta = beta; 
}

double
BDTLearner::frac_random_events () const {
    return m_frac_random_events; 
}

void
BDTLearner::frac_random_events (double n) {
    m_frac_random_events = n; 
}

int
BDTLearner::num_trees () const {
    return m_num_trees;
}

void
BDTLearner::num_trees (int n) {
    m_num_trees = n; 
}

bool
BDTLearner::use_purity () const {
    return m_use_purity;
}

void
BDTLearner::use_purity (bool purity) {
    m_use_purity = purity;
}

bool
BDTLearner::quiet () const {
    return m_quiet;
}

void
BDTLearner::quiet (bool val) {
    m_quiet = val;
}


boost::python::list
BDTLearner::after_pruners () const
{
    return np::vector_to_list (m_after_pruners);
}

boost::python::list
BDTLearner::before_pruners () const
{
    return np::vector_to_list (m_before_pruners);
}

void
BDTLearner::add_after_pruner (boost::shared_ptr<Pruner> pruner)
{
    m_after_pruners.push_back (pruner);
}

void
BDTLearner::add_before_pruner (boost::shared_ptr<Pruner> pruner)
{
    m_before_pruners.push_back (pruner);
}

void
BDTLearner::clear_after_pruners ()
{
    m_after_pruners.clear ();
}

void
BDTLearner::clear_before_pruners ()
{
    m_before_pruners.clear ();
}


void
BDTLearner::set_defaults ()
{
    m_dtlearner = boost::make_shared<DTLearner> (
        m_feature_names, m_sig_weight_name, m_bg_weight_name);

    m_beta = 1;
    m_frac_random_events = 1.;
    m_num_trees = 300;
    m_use_purity = false;
    m_quiet = false;

    clear_after_pruners ();
    clear_before_pruners ();
}


boost::shared_ptr<Model>
BDTLearner::train_given_everything (
    const vector<Event>& all_sig_events, const vector<Event>& all_bg_events,
    const vector<double>& init_sig_weights,
    const vector<double>& init_bg_weights) const
{
    typedef vector<Event> vecev;
    typedef vector<double> vecd;
    typedef vecev::const_iterator vecev_citer;
    typedef vecd::const_iterator vecd_citer;
    typedef vecd::iterator vecd_iter;
    using namespace np;

    int n_sig (all_sig_events.size ());
    int n_bg (all_bg_events.size ());
    int N_f (all_sig_events[0].size ());
    vector<double> all_sig_weights (
        div (init_sig_weights, sum (init_sig_weights)));
    vector<double> all_bg_weights (
        div (init_bg_weights, sum (init_bg_weights)));

    // Warning: technically, this usage of dtl.min_split is not threadsafe
    DTLearner& dtl (*m_dtlearner);
    int save_min_split (m_dtlearner->m_min_split);
    dtl.min_split (max (
        dtl.min_split (),
        static_cast<int> (1.0 * (n_sig + n_bg) / N_f / N_f / 20)));

    vector<boost::shared_ptr<DTModel> > dtmodels;
    vector<double> errs;
    vector<double> alphas;

    Notifier<int> notifier ("training decision trees", m_num_trees);
    if (not m_quiet) {
        notifier.update (0);
    }
    for (int m = 0; m < m_num_trees; ++m) {
        const int n_sig_used = static_cast<int>((m_frac_random_events * n_sig));
        const int n_bg_used = static_cast<int> ((m_frac_random_events * n_bg));
        const int n_sig_unused = (n_sig - n_sig_used);
        const int n_bg_unused = (n_bg - n_bg_used);
        // storage for picked events, if not using all events
        vector<Event> picked_sig_events;
        vector<double> picked_sig_weights;
        vector<Event> picked_bg_events;
        vector<double> picked_bg_weights;
        const vector<Event>* sig_events;
        const vector<double>* sig_weights;
        const vector<Event>* bg_events;
        const vector<double>* bg_weights;
        // if desired, pick events
        if (n_sig_unused > 0 or n_bg_unused > 0) {
            const vector<int> sig_indices (
                dtl.m_random_sampler.sample_range<int> (
                    n_sig_used, 0, n_sig, true));
            picked_sig_events =
                np::subscript (all_sig_events, sig_indices);
            picked_sig_weights =
                np::subscript (all_sig_weights, sig_indices);
            const vector<int> bg_indices (
                dtl.m_random_sampler.sample_range<int> (
                    n_bg_used, 0, n_bg, true));
            picked_bg_events =
                np::subscript (all_bg_events, bg_indices);
            picked_bg_weights =
                np::subscript (all_bg_weights, bg_indices);
            sig_events = &picked_sig_events;
            sig_weights = &picked_sig_weights;
            bg_events = &picked_bg_events;
            bg_weights = &picked_bg_weights;
        }
        else {
            sig_events = &all_sig_events;
            sig_weights = &all_sig_weights;
            bg_events = &all_bg_events;
            bg_weights = &all_bg_weights;
        }
        // note: per-node purity set from subsample, not all training events
        boost::shared_ptr<Model> model (
            dtl.train_given_everything (
                *sig_events, *bg_events, *sig_weights, *bg_weights));
        boost::shared_ptr<DTModel> dtmodel (
            dynamic_pointer_cast<DTModel> (model));
        dtmodels.push_back (dtmodel);


        // before-pruners get to prune before boosting
        for (vector<boost::shared_ptr<Pruner> >::const_iterator i_pruner
             = m_before_pruners.begin (); i_pruner != m_before_pruners.end ();
             ++i_pruner) {
            (*i_pruner)->prune (dtmodel);
        }

        // get scores for boosting
        const vector<double> all_sig_result (
            model->score (all_sig_events, m_use_purity, true));
        const vector<double> all_bg_result (
            model->score (all_bg_events, m_use_purity, true));

        // AdaBoost: find sum of weights for misclass'd events
        vecev_citer i_ev;
        vecd_iter i_weight;
        vecd_citer i_result;
        double sum_sig_weights (0);
        double sum_wrong_sig_weights (0);
        for (i_weight = all_sig_weights.begin (),
             i_result = all_sig_result.begin();
             i_weight != all_sig_weights.end (); ++i_weight, ++i_result) {
            sum_sig_weights += *i_weight;
            if (*i_result < 0) {
                sum_wrong_sig_weights += *i_weight;
            }
        }
        double sum_bg_weights (0);
        double sum_wrong_bg_weights (0);
        for (i_weight = all_bg_weights.begin (),
             i_result = all_bg_result.begin();
             i_weight != all_bg_weights.end (); ++i_weight, ++i_result) {
            sum_bg_weights += *i_weight;
            if (*i_result > 0) {
                sum_wrong_bg_weights += *i_weight;
            }
        }

        // AdaBoost: boost weights of misclass'd events
        // if (m_use_purity) then follow the SAMME.R algorithm from
        // J. Zhu, H. Zou, S. Rosset, T. Hastie.
        // "Multi-class AdaBoost", 2009.
        // https://web.stanford.edu/~hastie/Papers/samme.pdf
        const double err_m ((sum_wrong_sig_weights + sum_wrong_bg_weights)
                            / (sum_sig_weights + sum_bg_weights));
        const double boost_factor (pow ((1-err_m)/err_m, m_beta));
        const double alpha_m (m_beta ? log (boost_factor) : 1);
        const double EPS (Model::get_epsilon ());
        for (i_weight = all_sig_weights.begin (),
             i_result = all_sig_result.begin(),
             i_ev = all_sig_events.begin ();
             i_weight != all_sig_weights.end ();
             ++i_weight, ++i_result, ++i_ev) {
            const double result (*i_result);
            if (m_use_purity) {
                if (*i_weight < 0) {
                    continue;
                }
                const double p (min (1 - EPS, max (EPS, .5 * (result + 1))));
                const double booster (log (max (p, 1-p)) - log (min (p, 1-p)));
                *i_weight *= exp (-m_beta * booster);
            }
            else if (result < 0) {
                *i_weight *= boost_factor;
            }
        }
        for (i_weight = all_bg_weights.begin (),
             i_result = all_bg_result.begin(),
             i_ev = all_bg_events.begin ();
             i_weight != all_bg_weights.end ();
             ++i_weight, ++i_result, ++i_ev) {
            const double result (*i_result);
            if (m_use_purity) {
                if (*i_weight < 0) {
                    continue;
                }
                const double p (min (1 - EPS, max (EPS, .5 * (-result + 1))));
                const double booster (log (max (p, 1-p)) - log (min (p, 1-p)));
                *i_weight *= exp (-m_beta * booster);
            }
            else if (result > 0) {
                *i_weight *= boost_factor;
            }
        }
        errs.push_back (err_m);
        alphas.push_back (alpha_m);
        // Renormalize weights
        const double new_total_weight (
            sum (all_sig_weights) + sum (all_bg_weights));
        all_sig_weights = div (all_sig_weights, new_total_weight);
        all_bg_weights = div (all_bg_weights, new_total_weight);

        // after-pruners get to prune after boosting
        for (vector<boost::shared_ptr<Pruner> >::const_iterator i_pruner
             = m_after_pruners.begin (); i_pruner != m_after_pruners.end ();
             ++i_pruner) {
            (*i_pruner)->prune (dtmodel);
        }
        if (not m_quiet) {
            notifier.update (m + 1);
        }
    }
    if (not m_quiet) {
        notifier.finish ();
    }
    dtl.min_split (save_min_split);
    return boost::make_shared<BDTModel> (m_feature_names, dtmodels, alphas);
}

void
export_bdtlearner ()
{
    using namespace boost::python;

    class_<BDTLearner, bases<Learner> > (
        "BDTLearner",
        "Train a boosted decision tree.\n\n"
        "The feature_names, sig_weight_name and bg_weight_name are chosen\n"
        "at construction and are intrinsic to each BDTLearner.  After\n"
        "construction, various options can be set and one or more\n"
        "boosted decision trees can be be trained.  Each training\n"
        "results in a new Model object."
        ,init<boost::python::list>())
        .def (init<boost::python::list,string> ())
        .def (init<boost::python::list,string,string> ())
        .add_property (
            "dtlearner", &BDTLearner::dtlearner)
        .add_property (
            "beta",
            (double (BDTLearner::*)()const) &BDTLearner::beta,
            (void (BDTLearner::*)(double)) &BDTLearner::beta)
        .add_property (
            "frac_random_events",
            (double (BDTLearner::*)()const) &BDTLearner::frac_random_events,
            (void (BDTLearner::*)(double)) &BDTLearner::frac_random_events)
        .add_property (
            "num_trees", 
            (int (BDTLearner::*)()const) &BDTLearner::num_trees,
            (void (BDTLearner::*)(int)) &BDTLearner::num_trees)
        .add_property (
            "use_purity", 
            (bool (BDTLearner::*)()const) &BDTLearner::use_purity,
            (void (BDTLearner::*)(bool)) &BDTLearner::use_purity)
        .add_property (
            "quiet", 
            (bool (BDTLearner::*)()const) &BDTLearner::quiet,
            (void (BDTLearner::*)(bool)) &BDTLearner::quiet)
        .add_property (
            "after_pruners", &BDTLearner::after_pruners)
        .add_property (
            "before_pruners", &BDTLearner::before_pruners)
        .def ("add_after_pruner", &BDTLearner::add_after_pruner)
        .def ("add_before_pruner", &BDTLearner::add_before_pruner)
        .def ("clear_after_pruners", &BDTLearner::clear_after_pruners)
        .def ("clear_before_pruners", &BDTLearner::clear_before_pruners)
        .def ("set_defaults", &BDTLearner::set_defaults)
        ;

    register_ptr_to_python <boost::shared_ptr<BDTLearner> > ();
}
