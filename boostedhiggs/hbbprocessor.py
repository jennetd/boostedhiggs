import logging
import numpy as np
import awkward as ak
import json
import copy
from coffea import processor, hist
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from boostedhiggs.btag import BTagEfficiency, BTagCorrector
from boostedhiggs.common import (
    getBosons,
    bosonFlavor,
    pass_json_array,
)
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_jetTriggerWeight,
    jet_factory,
    fatjet_factory,
    add_jec_variables,
    met_factory,
)


logger = logging.getLogger(__name__)


def shifter(*path):
    def try_shift(objs):
        """Try to access a shifted collection, e.g. jets.JEC.up If not available, return the base collection"""
        try:
            return objs[path]
        except ValueError:
            # Why not KeyError?
            return objs

    return try_shift


class HbbProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='pt', v2=False, v3=False, v4=False):
        # v2 DDXv2
        # v3 ParticleNet
        # v4 mix
        self._year = year
        self._v2 = v2
        self._v3 = v3
        self._v4 = v4
        self._jet_arbitration = jet_arbitration

        self._btagSF = BTagCorrector(year, 'medium')

        self._msdSF = {
            '2016': 1.,
            '2017': 0.987,
            '2018': 0.970,
        }

        self._muontriggers = {
            '2016': [
                'Mu50',  # TODO: check
            ],
            '2017': [
                'Mu50',
                'TkMu50',
            ],
            '2018': [
                'Mu50',  # TODO: check
            ],
        }

        self._triggers = {
            '2016': [
                'PFHT800',
                'PFHT900',
                'AK8PFJet360_TrimMass30',
                'AK8PFHT700_TrimR0p1PT0p03Mass50',
                'PFHT650_WideJetMJJ950DEtaJJ1p5',
                'PFHT650_WideJetMJJ900DEtaJJ1p5',
                'AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                'PFJet450',
            ],
            '2017': [
                'AK8PFJet330_PFAK8BTagCSV_p17',
                'PFHT1050',
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFJet500',
                'AK8PFJet500',

            ],
            '2018': [
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFHT1050',
                'PFJet500',
                'AK8PFJet500',
                # 'AK8PFJet330_PFAK8BTagCSV_p17', not present in 2018D?
                'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
            ],
        }

        self._json_paths = {
            '2016': 'jsons/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt', 
            '2017': 'jsons/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt',
            '2018': 'jsons/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt',
        }

        self._accumulator = processor.dict_accumulator({
            # dataset -> sumw
            'sumw': processor.defaultdict_accumulator(float),
            'cutflow_msd': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('msd', r'Jet $m_{sd}$', 23, 40, 201),
            ),
            'cutflow_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('eta', r'Jet $\eta$', 40, -2.5, 2.5),
            ),
            'cutflow_pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', 100, 400, 1200),
            ),
            'nminus1_n2ddt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('n2ddt', 'N2ddt value', 40, -0.25, 0.25),
            ),
            'btagWeight': hist.Hist('Events', hist.Cat('dataset', 'Dataset'), hist.Bin('val', 'BTag correction', 50, 0, 3)),
            'templates': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('msd', r'Jet $m_{sd}$', 23, 40, 201),
                hist.Bin('ddb', r'Jet ddb score', [0, 0.89, 1]),
                hist.Bin('ddc', r'Jet ddc score', [0, 0.1, 0.44, .83, 1]),
                hist.Bin('ddcvb', r'Jet ddcvb score', [0, 0.017, 0.2, 1]),
            ),
            'signal_opt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('ddc', r'Jet ddc score', np.r_[0, np.geomspace(0.0001, 1, 101)]) if self._v2 else hist.Bin('ddc', r'Jet ddc score', 100, 0, 1), 
                hist.Bin('ddcvb', r'Jet ddcvb score',np.r_[0, np.geomspace(0.0001, 1, 101)]) if self._v2 else hist.Bin('ddcvb', r'Jet ddc score', 100, 0, 1), 
            ),
            'signal_optb': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('ddb', r'Jet ddb score', np.r_[0, np.geomspace(0.0001, 1, 101)]) if self._v2 else hist.Bin('ddb', r'Jet ddc score', 100, 0, 1), 
            ),
            'wtag_opt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]), #
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', 100, 200, 1200),
                #hist.Bin('msd', r'Jet $m_{sd}$', 23, 40, 201),
                hist.Bin('msd', r'Jet $m_{sd}$', 30, 40, 201),
                hist.Bin('n2ddt', 'N2ddt value', 20, -0.5, 0.5),
                hist.Bin('ddcvb', r'Jet ddcvb score', [0, 0.017, 0.2, 1]),
            ),
            'genresponse_noweight': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('genpt', r'Generated Higgs $p_{T}$ [GeV]', [200, 300, 450, 650, 7500]),
            ),
            'genresponse': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('genpt', r'Generated Higgs $p_{T}$ [GeV]', [200, 300, 450, 650, 7500]),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        output = self.accumulator.identity()
        jec_cache = {}
        output += self.process_shift(events, lambda objs: objs, None, jec_cache)
        output += self.process_shift(events, shifter("JES", "up"), "JESUp", jec_cache)
        output += self.process_shift(events, shifter("JES", "down"), "JESDown", jec_cache)
        output += self.process_shift(events, shifter("JER", "up"), "JERUp", jec_cache)
        output += self.process_shift(events, shifter("JER", "up"), "JERDown", jec_cache)
        output += self.process_shift(events, shifter("MET_UnclusteredEnergy", "up"), "UESUp", jec_cache)
        output += self.process_shift(events, shifter("MET_UnclusteredEnergy", "down"), "UESDown", jec_cache)
        return output

    def process_shift(self, events, shift_fn, shift_name, jec_cache):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        weights = Weights(len(events))
        weights_wtag = copy.deepcopy(weights)
        output = self.accumulator.identity()
        if not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            lumiMask = LumiMask(self._json_paths[self._year])
            lumi_mask = lumiMask(events.run, events.luminosityBlock)
            for t in self._triggers[self._year]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
            trigger = trigger & lumi_mask
            # print(f"Lumipass: {np.sum(lumi_mask)}/{len(lumi_mask)}")
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('trigger', trigger)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            lumiMask = LumiMask(self._json_paths[self._year])
            lumi_mask = lumiMask(events.run, events.luminosityBlock)
            for t in self._muontriggers[self._year]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
            trigger = trigger & lumi_mask
            # print(f"Lumipass: {np.sum(lumi_mask)}/{len(lumi_mask)}")
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('muontrigger', trigger)

        if isRealData:
            fatjets = events.FatJet
        else:
            fatjets = shift_fn(fatjet_factory[f"{self._year}mc"].build(add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache))
        
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['qcdrho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        fatjets['n2ddt'] = fatjets.n2b1 - n2ddt_shift(fatjets, year=self._year)
        fatjets['msdcorr_full'] = fatjets['msdcorr'] * self._msdSF[self._year]
        
        candidatejet = fatjets[
            # https://github.com/DAZSLE/BaconAnalyzer/blob/master/Analyzer/src/VJetLoader.cc#L269
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight  # this is loose in sampleContainer
        ]
        if self._jet_arbitration == 'pt':
            candidatejet = ak.firsts(candidatejet)
        elif self._jet_arbitration == 'mass':
            candidatejet = candidatejet[
                ak.argmax(candidatejet.msdcorr)
            ]
        elif self._jet_arbitration == 'n2':
            candidatejet = candidatejet[
                ak.argmin(candidatejet.n2ddt)
            ]
        elif self._jet_arbitration == 'ddb':
            candidatejet = candidatejet[
                ak.argmax(candidatejet.btagDDBvL)
            ]
        else:
            raise RuntimeError("Unknown candidate jet arbitration")
    

        selection.add('minjetkin',
            (candidatejet.pt >= 450)
            & (candidatejet.msdcorr >= 40.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('jetacceptance',
            (candidatejet.msdcorr >= 40.)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr < 201.)
        )
        selection.add('jetkinematics',
            (candidatejet.pt > 450)
        )
        selection.add('jetid', candidatejet.isTight)
        selection.add('n2ddt', (candidatejet.n2ddt < 0.))
        if not self._v2:
            selection.add('ddbpass', (candidatejet.btagDDBvL >= 0.89))
            selection.add('ddcpass', (candidatejet.btagDDCvL >= 0.83))
            selection.add('ddcvbpass', (candidatejet.btagDDCvB >= 0.2))
        else:
            selection.add('ddbpass', (candidatejet.btagDDBvLV2 >= 0.89))
            selection.add('ddcpass', (candidatejet.btagDDCvLV2 >= 0.83))
            selection.add('ddcvbpass', (candidatejet.btagDDCvBV2 >= 0.2))

        if isRealData:
            jets = events.Jet
            met = events.MET
        else:
            jets = shift_fn(jet_factory[f"{self._year}mc"].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache))
            met = shift_fn(met_factory.build(events.MET, jets, jec_cache))

        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 2.5)
            & jets.isTight
        ]
        # Protect again "empty" arrays [None, None, None...]
        # if ak.sum(candidatejet.phi) == 0.:
        #     return self.accumulator.identity()
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('antiak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) < BTagEfficiency.btagWPs[self._year]['medium'])
        ak4_away = jets[dphi > 0.8]
        selection.add('ak4btagMedium08', ak.max(ak4_away.btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])

        selection.add('met', met.pt < 140.)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        leadingmuon = ak.firsts(events.Muon[goodmuon])

        nelectrons = ak.sum(
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE),
            axis=1,
        )

        ntaus = ak.sum(
            (events.Tau.pt > 20)
            & events.Tau.idDecayMode,  # bacon iso looser than Nano selection
            axis=1,
        )

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        selection.add('muonkin', (leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        selection.add('muonDphiAK8', abs(leadingmuon.delta_phi(candidatejet)) > 2*np.pi/3)

        # W-Tag
        # tag side
        selection.add('ak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])
        selection.add('met40p', met.pt > 40.) 
        selection.add('tightMuon', (leadingmuon.tightId) & (leadingmuon.pt > 53.))
        selection.add('ptrecoW', (leadingmuon + met).pt > 250.)
        selection.add('ak4btagNearMu', leadingmuon.delta_r(leadingmuon.nearest(ak4_away, axis=None)) < 2.0 )
        # probe side
        selection.add('minWjetpteta', (candidatejet.pt >= 200) & (abs(candidatejet.eta) < 2.4))
        selection.add('noNearMuon', candidatejet.delta_r(candidatejet.nearest(events.Muon[goodmuon], axis=None)) > 1.0)
        #####

        if isRealData :
            genflavor = candidatejet.pt - candidatejet.pt  # zeros_like
        else:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            genflavor = bosonFlavor(matchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            weights_wtag = copy.deepcopy(weights)
            add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)
            output['btagWeight'].fill(dataset=dataset, val=self._btagSF.addBtagWeight(weights, ak4_away))
            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        msd_matched = candidatejet.msdcorr * self._msdSF[self._year] * (genflavor > 0) + candidatejet.msdcorr * (genflavor == 0)

        regions = {
            'signal': ['noleptons', 'minjetkin',  'met', 'jetid', 'antiak4btagMediumOppHem', 'n2ddt', 'trigger'],
            'signal_noddt': ['noleptons', 'minjetkin',  'met', 'jetid', 'antiak4btagMediumOppHem', 'trigger'],
            'muoncontrol': ['muontrigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8'],
            'muoncontrolCC': ['muontrigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8', 'ddcvbpass'],
            'wtag': ['muontrigger', 'minWjetpteta',  'ak4btagMediumOppHem', 'met40p', 'tightMuon', 'noNearMuon', 'ptrecoW', 'ak4btagNearMu'],
            'wtag0': ['minWjetpteta',  'ak4btagMediumOppHem', 'met40p', 'tightMuon', 'noNearMuon', 'ptrecoW', 'ak4btagNearMu'],
            'noselection': [],
        }

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        for region, cuts in regions.items():
            # allcuts = set(["id"])
            allcuts = set([])
            cut = selection.all(*allcuts)
            output['cutflow_msd'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, None),
                                   cut=0, weight=weights.weight(), msd=normalize(msd_matched, None))
            output['cutflow_eta'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                   cut=0, weight=weights.weight()[cut], eta=normalize(candidatejet.eta, cut))
            output['cutflow_pt'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                   cut=0, weight=weights.weight()[cut], pt=normalize(candidatejet.pt, cut))
            for i, cut in enumerate(cuts + ['ddcvbpass', 'ddcpass']):
                allcuts.add(cut)
                cut = selection.all(*allcuts)
                output['cutflow_msd'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                       cut=i + 1, weight=weights.weight()[cut], msd=normalize(msd_matched, cut))
                output['cutflow_eta'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                        cut=i + 1, weight=weights.weight()[cut], eta=normalize(candidatejet.eta, cut))
                output['cutflow_pt'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                        cut=i + 1, weight=weights.weight()[cut], pt=normalize(candidatejet.pt, cut))


        if shift_name is None:
            systematics = [
                None,
                'jet_triggerUp',
                'jet_triggerDown',
                'btagWeightUp',
                'btagWeightDown',
                'btagEffStatUp',
                'btagEffStatDown',
            ]
        else:
            systematics = [shift_name]

        def fill(region, systematic, wmod=None):
            selections = regions[region]
            cut = selection.all(*selections)
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * wmod[cut]
            if systematic is None:
                weight_wtag = weights_wtag.weight(modifier=systematic)[cut]
            
            
            if self._v2:
                bvl = candidatejet.btagDDBvLV2
                cvl = candidatejet.btagDDCvLV2
                cvb = candidatejet.btagDDCvBV2
            elif self._v3:
                bvl = candidatejet.particleNet_HbbvsQCD
                cvl = candidatejet.particleNet_HccvsQCD
                cvb = candidatejet.particleNetMD_Xcc/(candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)
            elif self._v4:
                bvl = candidatejet.particleNet_HbbvsQCD
                cvl = candidatejet.btagDDCvLV2
                cvb = candidatejet.particleNetMD_Xcc/(candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)
            else:
                bvl = candidatejet.btagDDBvL
                cvl = candidatejet.btagDDCvL
                cvb = candidatejet.btagDDCvB

            output['templates'].fill(
                dataset=dataset,
                region=region,
                systematic=sname,
                genflavor=normalize(genflavor, cut),
                pt=normalize(candidatejet.pt, cut),
                msd=normalize(msd_matched, cut),
                ddb=normalize(bvl, cut),
                ddc=normalize(cvl, cut),
                ddcvb=normalize(cvb, cut),
                weight=weight,
            )
            if systematic is None:
                output['wtag_opt'].fill(
                    dataset=dataset,
                    region=region,
                    systematic=sname,
                    genflavor=normalize(genflavor, cut),
                    pt=normalize(candidatejet.pt, cut),
                    msd=normalize(msd_matched, cut),
                    n2ddt=normalize(candidatejet.n2ddt, cut),
                    ddcvb=normalize(cvb, cut),
                    weight=weight_wtag,
                )
            if wmod is not None:
                output['genresponse_noweight'].fill(
                    dataset=dataset,
                    region=region,
                    systematic=sname,
                    pt=normalize(candidatejet.pt, cut),
                    genpt=normalize(genBosonPt, cut),
                    weight=events.genWeight[cut] * wmod[cut],
                )
                output['genresponse'].fill(
                    dataset=dataset,
                    region=region,
                    systematic=sname,
                    pt=normalize(candidatejet.pt, cut),
                    genpt=normalize(genBosonPt, cut),
                    weight=weight,
                )
            if systematic is None:
                output['signal_opt'].fill(
                    dataset=dataset,
                    region=region,
                    genflavor=normalize(genflavor, cut),
                    ddc=normalize(cvl, cut),
                    ddcvb=normalize(cvb, cut),
                    weight=weight,
                )
                output['signal_optb'].fill(
                    dataset=dataset,
                    region=region,
                    genflavor=normalize(genflavor, cut),
                    ddb=normalize(bvl, cut),
                    weight=weight,
                )


        for region in regions:
            cut = selection.all(*(set(regions[region]) - {'n2ddt'}))
            output['nminus1_n2ddt'].fill(
                dataset=dataset,
                region=region,
                n2ddt=normalize(candidatejet.n2ddt, cut),
                weight=weights.weight()[cut],
            )
            for systematic in systematics:
                if isRealData and systematic is not None:
                    continue
                fill(region, systematic)
            if shift_name is None and 'GluGluHToBB' in dataset and 'LHEWeight' in events.fields:
                for i in range(9):
                    fill(region, 'LHEScale_%d' % i, events.LHEScaleWeight[:, i])
                for c in events.LHEWeight.fields[1:]:
                    fill(region, 'LHEWeight_%s' % c, events.LHEWeight[c])

        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
